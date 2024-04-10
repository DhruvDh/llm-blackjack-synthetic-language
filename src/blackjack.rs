use std::collections::HashMap;

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

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct HandValue {
    soft: usize,
    hard: usize,
}

impl HandValue {
    fn get_score(&self, target_score: usize) -> usize {
        if self.soft > target_score && self.hard > target_score {
            self.soft.min(self.hard)
        } else if self.soft <= target_score && self.hard <= target_score {
            self.soft.max(self.hard)
        } else if self.soft <= target_score {
            self.soft
        } else {
            self.hard
        }
    }
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
        self.cards.pop()
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
        total_value: HandValue,
    },
    Probabilities {
        player:      Player,
        improvement: String,
        bust:        String,
    },
    Busts {
        player: Player,
        score:  usize,
    },
    Wins {
        player:       Player,
        user_score:   usize,
        dealer_score: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
enum ActionEvent {
    Hits { player: Player, card: Card },
    Stands { player: Player },
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
    GameTied {
        game_id:      usize,
        user_score:   usize,
        dealer_score: usize,
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
        if (*count) > 0 {
            *count -= 1;
        }
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
        self.deck.card_counter.update(self.player_hand[0]);
        self.deck.card_counter.update(self.player_hand[1]);
        events.push(Event::Info {
            game_id: self.game.game_id,
            event:   InfoEvent::Hand {
                player:      Player::Dealer,
                cards:       vec![self.dealer_hand[0]],
                total_value: self.hand_value(&[self.dealer_hand[0]]),
            },
        });
        self.deck.card_counter.update(self.dealer_hand[0]);

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
                    event:   ActionEvent::Hits {
                        player: Player::User,
                        card,
                    },
                });
                self.player_hand.push(card);
                self.deck.card_counter.update(card);
            } else {
                events.push(Event::GameTied {
                    game_id:      self.game.game_id,
                    user_score:   0,
                    dealer_score: 0,
                });
                return events;
            }

            let hand_values = self.hand_value(&self.player_hand);
            if hand_values.get_score(self.game.target_score) >= self.game.target_score {
                break;
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
            event:   ActionEvent::Stands {
                player: Player::User,
            },
        });
        events.push(Event::Info {
            game_id: self.game.game_id,
            event:   InfoEvent::Hand {
                player:      Player::User,
                cards:       self.player_hand.clone(),
                total_value: self.hand_value(&self.player_hand),
            },
        });

        let player_hand = self.hand_value(&self.player_hand);
        let mut dealer_hand = self.hand_value(&self.dealer_hand);

        while dealer_hand.soft < self.game.dealer_threshold
            || dealer_hand.hard < self.game.dealer_threshold
        {
            if dealer_hand.get_score(self.game.target_score) == self.game.target_score {
                break;
            }

            if let Some(card) = self.deck.draw() {
                events.push(Event::Action {
                    game_id: self.game.game_id,
                    event:   ActionEvent::Hits {
                        player: Player::Dealer,
                        card,
                    },
                });
                self.dealer_hand.push(card);
                self.deck.card_counter.update(card);

                dealer_hand = self.hand_value(&self.dealer_hand);
            } else {
                events.push(Event::GameTied {
                    game_id:      self.game.game_id,
                    user_score:   0,
                    dealer_score: 0,
                });
                return events;
            }
        }

        events.push(Event::Action {
            game_id: self.game.game_id,
            event:   ActionEvent::Stands {
                player: Player::Dealer,
            },
        });
        events.push(Event::Info {
            game_id: self.game.game_id,
            event:   InfoEvent::Hand {
                player:      Player::Dealer,
                cards:       self.dealer_hand.clone(),
                total_value: self.hand_value(&self.dealer_hand),
            },
        });

        let dealer_hand = self.hand_value(&self.dealer_hand);
        let user_score = player_hand.get_score(self.game.target_score);
        let dealer_score = dealer_hand.get_score(self.game.target_score);

        if dealer_score > self.game.target_score && user_score > self.game.target_score {
            events.push(Event::Info {
                game_id: self.game.game_id,
                event:   InfoEvent::Busts {
                    player: Player::Dealer,
                    score:  dealer_score,
                },
            });
            events.push(Event::Info {
                game_id: self.game.game_id,
                event:   InfoEvent::Busts {
                    player: Player::User,
                    score:  user_score,
                },
            });
            events.push(Event::GameTied {
                game_id: self.game.game_id,
                user_score,
                dealer_score,
            });
        } else if user_score > self.game.target_score {
            events.push(Event::Info {
                game_id: self.game.game_id,
                event:   InfoEvent::Busts {
                    player: Player::User,
                    score:  user_score,
                },
            });
            events.push(Event::Info {
                game_id: self.game.game_id,
                event:   InfoEvent::Wins {
                    player: Player::Dealer,
                    user_score,
                    dealer_score,
                },
            });
        } else if dealer_score > self.game.target_score {
            events.push(Event::Info {
                game_id: self.game.game_id,
                event:   InfoEvent::Busts {
                    player: Player::Dealer,
                    score:  dealer_score,
                },
            });
            events.push(Event::Info {
                game_id: self.game.game_id,
                event:   InfoEvent::Wins {
                    player: Player::User,
                    user_score,
                    dealer_score,
                },
            });
        } else if user_score == self.game.target_score || user_score > dealer_score {
            events.push(Event::Info {
                game_id: self.game.game_id,
                event:   InfoEvent::Wins {
                    player: Player::User,
                    user_score,
                    dealer_score,
                },
            });
        } else if dealer_score == self.game.target_score || dealer_score > user_score {
            events.push(Event::Info {
                game_id: self.game.game_id,
                event:   InfoEvent::Wins {
                    player: Player::Dealer,
                    user_score,
                    dealer_score,
                },
            });
        } else {
            events.push(Event::GameTied {
                game_id: self.game.game_id,
                user_score,
                dealer_score,
            });
        }

        events
    }

    fn hand_value(&self, cards: &[Card]) -> HandValue {
        let (total, aces) = cards.iter().fold((0, 0), |(total, aces), card| {
            if card.value == 1 {
                (total + 1, aces + 1)
            } else {
                (total + card.value, aces)
            }
        });

        if aces > 0 {
            HandValue {
                soft: total,
                hard: total + 10,
            }
        } else {
            HandValue {
                soft: total,
                hard: total,
            }
        }
    }

    fn calculate_probabilities(&self) -> (f64, f64) {
        let remaining_cards = self.deck.card_counter.counts.values().sum::<usize>();
        let mut bust_prob = 0.0;
        let mut player_improvement_prob = 0.0;

        for card_value in 1..=self.deck.num_cards_per_suit {
            let prob = self.deck.card_counter.get_count(card_value) as f64 / remaining_cards as f64;

            let new_player_hand = [&self.player_hand[..], &[Card {
                suit:  0,
                value: card_value,
            }]]
            .concat();
            let hand_value = self.hand_value(&new_player_hand);

            if hand_value.get_score(self.game.target_score) > self.game.target_score {
                bust_prob += prob;
            } else {
                player_improvement_prob += prob;
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

pub fn generate_transcripts(
    num_suits: usize,
    num_cards_per_suit: usize,
    num_decks: usize,
    num_simultaenous_games: usize,
) -> String {
    let mut all_events = Vec::new();
    for game_id in 0..num_simultaenous_games {
        let game = Game::new(game_id, num_suits, num_cards_per_suit, num_decks);
        let events = game.generate_data();
        all_events.push(events);
    }

    let mut event_indices = vec![0; num_simultaenous_games];
    let mut all_done = false;
    let mut interleaved_events = Vec::new();

    while !all_done {
        all_done = true;
        for (game_index, events) in all_events.iter().enumerate() {
            if event_indices[game_index] < events.len() {
                all_done = false;
                let event = &events[event_indices[game_index]];
                let mut event_string = format!("{:?}", event);

                // Insert spaces before each digit in the event string
                event_string = event_string.chars().fold(String::new(), |mut acc, ch| {
                    if ch.is_ascii_digit() || ch == '*' || ch == '"' {
                        acc.push(' ');
                    }
                    acc.push(ch);
                    acc
                });

                interleaved_events.push(event_string);
                event_indices[game_index] += 1;
            }
        }
    }

    interleaved_events.join("\n")
}
