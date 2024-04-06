use std::collections::HashMap;

use bpaf::Bpaf;
use futures::{
    stream::FuturesUnordered,
    StreamExt,
};
use rand::{
    seq::SliceRandom,
    SeedableRng,
};
use rand_chacha::ChaCha12Rng;
use serde_derive::{
    Deserialize,
    Serialize,
};
use thiserror::Error;
use tokio::{
    fs::OpenOptions,
    io::AsyncWriteExt,
};

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
struct Card {
    suit:  usize,
    value: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Deck {
    cards:              Vec<Card>,
    num_suits:          usize,
    num_cards_per_suit: usize,
    num_decks:          usize,
    card_counter:       CardCounter,
}

impl Deck {
    fn new(num_suits: usize, num_cards_per_suit: usize, num_decks: usize) -> Self {
        let mut cards = Vec::new();
        for _ in 0..num_decks {
            for suit in 0..num_suits {
                for value in 1..=num_cards_per_suit {
                    cards.push(Card {
                        suit,
                        value,
                    });
                }
            }
        }
        Self {
            cards,
            num_suits,
            num_cards_per_suit,
            num_decks,
            card_counter: CardCounter::new(num_suits, num_cards_per_suit, num_decks),
        }
    }

    fn shuffle(&mut self) {
        let mut rng = ChaCha12Rng::from_entropy();
        self.cards.shuffle(&mut rng);
    }

    fn draw(&mut self) -> Option<Card> {
        self.cards.pop().map(|card| {
            self.card_counter.update(card);
            card
        })
    }

    fn remaining_cards(&self) -> usize {
        self.cards.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
enum Player {
    User,
    Dealer,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
enum InfoEvent {
    Hand {
        player:      Player,
        cards:       Vec<Card>,
        total_value: (usize, usize),
    },
    Probabilities {
        player:      Player,
        improvement: String,
        bust:        String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
enum ActionEvent {
    Hits(Player, Card),
    Stands(Player),
    Busts(Player),
    Wins(Player),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
enum Event {
    GameStart {
        game_id:            usize,
        num_suits:          usize,
        num_cards_per_suit: usize,
        num_decks:          usize,
        target_score:       usize,
        dealer_threshold:   usize,
    },
    RoundStart {
        game_id:  usize,
        round_id: usize,
    },
    Info {
        game_id: usize,
        event:   InfoEvent,
    },
    Action {
        game_id: usize,
        event:   ActionEvent,
    },
    Tie {
        game_id: usize,
    },
    GameTied {
        game_id: usize,
    },
}

#[derive(Debug, Clone)]
struct CardCounter {
    counts: HashMap<usize, usize>,
}

impl CardCounter {
    fn new(num_suits: usize, num_cards_per_suit: usize, num_decks: usize) -> Self {
        let mut counts = HashMap::with_capacity(num_cards_per_suit);
        for value in 1..=num_cards_per_suit {
            counts.insert(value, num_decks * num_suits);
        }
        Self {
            counts,
        }
    }

    fn update(&mut self, card: Card) {
        let count = self.counts.get_mut(&card.value).unwrap();
        *count -= 1;
    }

    fn get_count(&self, value: usize) -> usize {
        *self.counts.get(&value).unwrap()
    }
}

#[derive(Error, Debug)]
enum BlackjackError {
    #[error("No cards remaining in the deck.")]
    NoCardsRemaining,
}

fn probability_to_stars(probability: f64) -> String {
    let num_stars = (probability * 10.0).round() as usize;
    "*".repeat(num_stars)
}

struct Round<'a> {
    game:        Game,
    player_hand: Vec<Card>,
    dealer_hand: Vec<Card>,
    deck:        &'a mut Deck,
}

impl<'a> Round<'a> {
    fn new(game: Game, deck: &'a mut Deck) -> Self {
        let player_hand: Vec<Card> = (0..2).filter_map(|_| deck.draw()).collect();
        let dealer_hand: Vec<Card> = (0..2).filter_map(|_| deck.draw()).collect();

        Self {
            game,
            player_hand,
            dealer_hand,
            deck,
        }
    }

    fn play(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        events.push(Event::Info {
            game_id: self.game.game_id,
            event:   InfoEvent::Hand {
                player:      Player::User,
                cards:       self.player_hand.clone(),
                total_value: self.hand_value(&self.player_hand),
            },
        });
        // self.deck.card_counter.update(self.player_hand[0]);
        // self.deck.card_counter.update(self.player_hand[1]);
        events.push(Event::Info {
            game_id: self.game.game_id,
            event:   InfoEvent::Hand {
                player:      Player::Dealer,
                cards:       vec![self.dealer_hand[0]],
                total_value: self.hand_value(&[self.dealer_hand[0]]),
            },
        });
        // self.deck.card_counter.update(self.dealer_hand[0]);

        while self.player_should_hit().is_ok_and(|should_hit| should_hit) {
            if let Some(card) = self.deck.draw() {
                let (bust_prob, player_improvement_prob) = self.calculate_probabilities();
                events.push(Event::Info {
                    game_id: self.game.game_id,
                    event:   InfoEvent::Probabilities {
                        player:      Player::User,
                        improvement: probability_to_stars(player_improvement_prob),
                        bust:        probability_to_stars(bust_prob),
                    },
                });

                events.push(Event::Action {
                    game_id: self.game.game_id,
                    event:   ActionEvent::Hits(Player::User, card),
                });
                self.player_hand.push(card);
                // self.deck.card_counter.update(card);
            } else {
                events.push(Event::GameTied {
                    game_id: self.game.game_id,
                });
                return events;
            }
        }

        let (bust_prob, player_improvement_prob) = self.calculate_probabilities();
        events.push(Event::Info {
            game_id: self.game.game_id,
            event:   InfoEvent::Probabilities {
                player:      Player::User,
                improvement: probability_to_stars(player_improvement_prob),
                bust:        probability_to_stars(bust_prob),
            },
        });
        events.push(Event::Action {
            game_id: self.game.game_id,
            event:   ActionEvent::Stands(Player::User),
        });
        events.push(Event::Info {
            game_id: self.game.game_id,
            event:   InfoEvent::Hand {
                player:      Player::User,
                cards:       self.player_hand.clone(),
                total_value: self.hand_value(&self.player_hand),
            },
        });

        let (_, player_hard_value) = self.hand_value(&self.player_hand);
        if player_hard_value > self.game.target_score {
            events.push(Event::Action {
                game_id: self.game.game_id,
                event:   ActionEvent::Busts(Player::User),
            });
            events.push(Event::Action {
                game_id: self.game.game_id,
                event:   ActionEvent::Wins(Player::Dealer),
            });
        } else {
            while {
                let (_, dealer_hard_value) = self.hand_value(&self.dealer_hand);
                dealer_hard_value < self.game.dealer_threshold
            } {
                if let Some(card) = self.deck.draw() {
                    events.push(Event::Action {
                        game_id: self.game.game_id,
                        event:   ActionEvent::Hits(Player::Dealer, card),
                    });
                    self.dealer_hand.push(card);
                    // self.deck.card_counter.update(card);
                } else {
                    events.push(Event::GameTied {
                        game_id: self.game.game_id,
                    });
                    return events;
                }
            }

            events.push(Event::Action {
                game_id: self.game.game_id,
                event:   ActionEvent::Stands(Player::Dealer),
            });
            events.push(Event::Info {
                game_id: self.game.game_id,
                event:   InfoEvent::Hand {
                    player:      Player::Dealer,
                    cards:       self.dealer_hand.clone(),
                    total_value: self.hand_value(&self.dealer_hand),
                },
            });

            let (player_soft_value, player_hard_value) = self.hand_value(&self.player_hand);
            let (dealer_soft_value, dealer_hard_value) = self.hand_value(&self.dealer_hand);

            let (player_score, dealer_score) = if player_soft_value <= self.game.target_score {
                (player_soft_value, dealer_soft_value)
            } else {
                (player_hard_value, dealer_hard_value)
            };

            if dealer_hard_value > self.game.target_score {
                events.push(Event::Action {
                    game_id: self.game.game_id,
                    event:   ActionEvent::Busts(Player::Dealer),
                });
                events.push(Event::Action {
                    game_id: self.game.game_id,
                    event:   ActionEvent::Wins(Player::User),
                });
            } else if player_score > dealer_score {
                events.push(Event::Action {
                    game_id: self.game.game_id,
                    event:   ActionEvent::Wins(Player::User),
                });
            } else if dealer_score > player_score {
                events.push(Event::Action {
                    game_id: self.game.game_id,
                    event:   ActionEvent::Wins(Player::Dealer),
                });
            } else {
                events.push(Event::Tie {
                    game_id: self.game.game_id,
                });
            }
        }

        events
    }

    fn hand_value(&self, cards: &[Card]) -> (usize, usize) {
        let (total, aces) = cards.iter().fold((0, 0), |(total, aces), card| {
            if card.value == 1 {
                (total + 1, aces + 1)
            } else {
                (total + card.value, aces)
            }
        });

        if aces > 0 {
            if total + 10 <= self.game.target_score {
                (total + 10, total)
            } else {
                (total, total)
            }
        } else {
            (total, total)
        }
    }

    fn calculate_probabilities(&self) -> (f64, f64) {
        let remaining_cards = self.deck.remaining_cards();
        let mut bust_prob = 0.0;
        let mut player_improvement_prob = 0.0;

        for card_value in 1..=self.deck.num_cards_per_suit {
            let prob = self.deck.card_counter.get_count(card_value) as f64 / remaining_cards as f64;

            let new_player_hand = [&self.player_hand[..], &[Card {
                suit:  0,
                value: card_value,
            }]]
            .concat();
            let (total_soft_value, total_hard_value) = self.hand_value(&new_player_hand);

            if total_soft_value > self.game.target_score {
                bust_prob += prob;
            } else if total_hard_value <= self.game.target_score {
                player_improvement_prob += prob;
            } else {
                bust_prob += prob;
            }
        }

        (bust_prob, player_improvement_prob)
    }

    fn player_should_hit(&self) -> Result<bool, BlackjackError> {
        let remaining_cards = self.deck.remaining_cards();

        if remaining_cards == 0 {
            return Err(BlackjackError::NoCardsRemaining);
        }

        let (bust_prob, player_improvement_prob) = self.calculate_probabilities();
        Ok(player_improvement_prob > bust_prob)
    }
}

#[derive(Debug, Clone, Copy)]
struct Game {
    game_id:            usize,
    num_suits:          usize,
    num_cards_per_suit: usize,
    num_decks:          usize,
    target_score:       usize,
    dealer_threshold:   usize,
}

fn calculate_thresholds(num_suits: usize, num_cards_per_suit: usize) -> (usize, usize) {
    let num_cards = num_suits * num_cards_per_suit;
    let max_card_value = num_cards_per_suit.min(10);
    let num_face_cards = num_suits * (num_cards_per_suit.saturating_sub(10));
    let num_aces = num_suits;

    let total_value: usize =
        (2..=max_card_value).sum::<usize>() * num_suits + num_face_cards * 10 + num_aces * 11;
    let avg_card_value = total_value as f64 / num_cards as f64;

    let target_score = (3.0 * avg_card_value).round() as usize;
    let dealer_threshold = (2.5 * avg_card_value).round() as usize;

    (target_score, dealer_threshold)
}

impl Game {
    fn new(game_id: usize, num_suits: usize, num_cards_per_suit: usize, num_decks: usize) -> Self {
        let (target_score, dealer_threshold) = calculate_thresholds(num_suits, num_cards_per_suit);
        Self {
            game_id,
            num_suits,
            num_cards_per_suit,
            num_decks,
            target_score,
            dealer_threshold,
        }
    }

    fn generate_data(&self) -> Vec<Event> {
        let mut deck = Deck::new(self.num_suits, self.num_cards_per_suit, self.num_decks);
        deck.shuffle();

        let mut events = Vec::new();
        events.push(Event::GameStart {
            game_id:            self.game_id,
            num_suits:          self.num_suits,
            num_cards_per_suit: self.num_cards_per_suit,
            num_decks:          self.num_decks,
            target_score:       self.target_score,
            dealer_threshold:   self.dealer_threshold,
        });

        let mut round_id = 0;
        while deck.remaining_cards() > 4 {
            events.push(Event::RoundStart {
                game_id: self.game_id,
                round_id,
            });

            let mut round = Round::new(*self, &mut deck);
            let round_events = round.play();
            events.extend(round_events);

            round_id += 1;
        }

        events
    }
}

fn generate_transcripts(
    num_suits: usize,
    num_cards_per_suit: usize,
    num_decks: usize,
    num_simultaneous_games: usize,
) -> String {
    let mut all_events = Vec::new();
    for game_id in 0..num_simultaneous_games {
        let game = Game::new(game_id, num_suits, num_cards_per_suit, num_decks);
        let events = game.generate_data();
        all_events.push(events);
    }

    let mut event_indices = vec![0; num_simultaneous_games];
    let mut all_done = false;
    let mut interleaved_events = Vec::new();

    while !all_done {
        all_done = true;
        for (game_index, events) in all_events.iter().enumerate() {
            if event_indices[game_index] < events.len() {
                all_done = false;
                let event = &events[event_indices[game_index]];
                interleaved_events.push(format!("{:?}", event));
                event_indices[game_index] += 1;
            }
        }
    }

    interleaved_events.join("\n")
}

#[derive(Clone, Debug, Bpaf)]
#[bpaf(options, version)]
struct Config {
    /// Minimum number of decks (default: 1)
    #[bpaf(long, fallback(1))]
    min_decks:                  usize,
    /// Maximum number of decks (default: 4)
    #[bpaf(long, fallback(4))]
    max_decks:                  usize,
    /// Minimum number of suits (default: 2)
    #[bpaf(long, fallback(2))]
    min_suits:                  usize,
    /// Maximum number of suits (default: 4)
    #[bpaf(long, fallback(4))]
    max_suits:                  usize,
    /// Minimum number of cards per suit (default: 6)
    #[bpaf(long, fallback(6))]
    min_cards_per_suit:         usize,
    /// Maximum number of cards per suit (default: 24)
    #[bpaf(long, fallback(20))]
    max_cards_per_suit:         usize,
    /// Number of games to generate for each configuration (default: 1)
    #[bpaf(short, long, fallback(100))]
    num_games:                  usize,
    /// Maximum number of simultaneous games to generate (default: 1)
    #[bpaf(short, long, fallback(1))]
    max_num_simultaneous_games: usize,
}

#[tokio::main]
async fn main() {
    let config = config().run();

    let file_path = "blackjack_transcripts.txt";

    let mut tasks = FuturesUnordered::new();

    for num_decks in config.min_decks..=config.max_decks {
        for num_suits in config.min_suits..=config.max_suits {
            for num_cards_per_suit in config.min_cards_per_suit..=config.max_cards_per_suit {
                for num_simultaenous_games in 1..=config.max_num_simultaneous_games {
                    for _ in 0..config.num_games {
                        let task = tokio::spawn(async move {
                            generate_transcripts(
                                num_suits,
                                num_cards_per_suit,
                                num_decks,
                                num_simultaenous_games,
                            )
                        });
                        tasks.push(task);
                    }
                }
            }
        }
    }

    let mut output = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)
        .await
        .expect("Unable to open file");

    let progress = indicatif::ProgressBar::new(tasks.len() as u64);
    while let Some(result) = tasks.next().await {
        let transcript = result.expect("Task panicked");
        output
            .write_all(transcript.as_bytes())
            .await
            .expect("Unable to write to file");
        output
            .write_all(b"\n")
            .await
            .expect("Unable to write to file");

        progress.inc(1);
    }

    let contents = tokio::fs::read_to_string(file_path)
        .await
        .expect("Unable to read file");

    let user_wins = contents.matches("Wins(User)").count();
    let dealer_wins = contents.matches("Wins(Dealer)").count();
    let ties = contents.matches("Tie {").count()
        + contents
            .matches(
                "GameTied
    {",
            )
            .count();
    let total_games = user_wins + dealer_wins + ties;

    println!(
        "User wins: {} ({:.2}%)",
        user_wins,
        user_wins as f64 / total_games as f64 * 100.0
    );
    println!(
        "Dealer wins: {} ({:.2}%)",
        dealer_wins,
        dealer_wins as f64 / total_games as f64 * 100.0
    );
    println!(
        "Ties: {} ({:.2}%)",
        ties,
        ties as f64 / total_games as f64 * 100.0
    );
}
