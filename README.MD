## Usage

```
Usage: generate-data [--min-decks=ARG] [--max-decks=ARG] [--min-suits=ARG] [--max-suits=ARG] [
--min-cards-per-suit=ARG] [--max-cards-per-suit=ARG] [-n=ARG] [-m=ARG]

Available options:
        --min-decks=ARG  Minimum number of decks (default: 1)
        --max-decks=ARG  Maximum number of decks (default: 4)
        --min-suits=ARG  Minimum number of suits (default: 2)
        --max-suits=ARG  Maximum number of suits (default: 4)
        --min-cards-per-suit=ARG  Minimum number of cards per suit (default: 6)
        --max-cards-per-suit=ARG  Maximum number of cards per suit (default: 24)
    -n, --num-games=ARG  Number of games to generate for each configuration (default: 1)
    -m, --max-num-simultaneous-games=ARG  Maximum number of simultaneous games to generate (default:
                         1)
    -h, --help           Prints help information
    -V, --version        Prints version information

```

## Information

> Note: This is a variant of Blackjack, and standard rules do not apply. Win rates should not be compared to standard Blackjack.

For implementation:

```rs
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
```

The winrates are:

```
User wins: 1339 (43.99%)
Dealer wins: 1147 (37.68%)
Ties: 558 (18.33%)
```

## Example

```sh
cargo run --release -- \
    -n 1 -m 1 \
    --min-decks=4 \
    --max-decks=4 \
    --min-suits=4 \
    --max-suits=4 \
    --min-cards-per-suit=13 \
    --max-cards-per-suit=13
```

```
    Finished `release` profile [optimized] target(s) in 0.07s
     Running `target/release/generate-data -n 1 -m 1 --min-decks=4 --max-decks=4 --min-suits=4 --max-suits=4 --min-cards-per-suit=13 --max-cards-per-suit=13`
    
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 1/1
User wins: 12 (36.36%)
Dealer wins: 13 (39.39%)
Ties: 8 (24.24%)
```

## Generated Data

Average number of tokens per game: ~620 tokens

```
User wins:        393,842 (43.16%)
Dealer wins:      344,935 (37.80%)
Ties:             173,694 (19.04%)
Total games:      912,471
Number of tokens: 563,086,839
```

```
GameStart { game_id:  0, num_suits:  4, num_cards_per_suit:  1 3, num_decks:  4, target_score:  2 2, dealer_threshold:  1 8 }
RoundStart { game_id:  0, round_id:  0 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  4 }, Card { suit:  1, value:  1 3 }], total_value: HandValue { soft:  1 7, hard:  1 7 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  5 }], total_value: HandValue { soft:  5, hard:  5 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * ", bust:  " * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  4 }, Card { suit:  1, value:  1 3 }], total_value: HandValue { soft:  1 7, hard:  1 7 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  5 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  3 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  1 2 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  5 }, Card { suit:  2, value:  1 }, Card { suit:  2, value:  5 }, Card { suit:  2, value:  3 }, Card { suit:  2, value:  1 2 }], total_value: HandValue { soft:  2 6, hard:  3 6 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 6 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 7, dealer_score:  2 6 } }
RoundStart { game_id:  0, round_id:  1 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  9 }, Card { suit:  0, value:  1 }], total_value: HandValue { soft:  1 0, hard:  2 0 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  2 }], total_value: HandValue { soft:  2, hard:  2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * * * ", bust:  " * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  0, value:  7 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * ", bust:  " * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  9 }, Card { suit:  0, value:  1 }, Card { suit:  0, value:  7 }], total_value: HandValue { soft:  1 7, hard:  2 7 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  6 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  3, value:  1 2 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  2 }, Card { suit:  0, value:  7 }, Card { suit:  0, value:  6 }, Card { suit:  3, value:  1 2 }], total_value: HandValue { soft:  2 7, hard:  2 7 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 7 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 7, dealer_score:  2 7 } }
RoundStart { game_id:  0, round_id:  2 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  1 0 }, Card { suit:  3, value:  1 1 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  9 }], total_value: HandValue { soft:  9, hard:  9 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * ", bust:  " * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  1 0 }, Card { suit:  3, value:  1 1 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  9 }, Card { suit:  0, value:  1 0 }], total_value: HandValue { soft:  1 9, hard:  1 9 } } }
Info { game_id:  0, event: Wins { player: User, user_score:  2 1, dealer_score:  1 9 } }
RoundStart { game_id:  0, round_id:  3 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  1 0 }, Card { suit:  3, value:  1 3 }], total_value: HandValue { soft:  2 3, hard:  2 3 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  2 }], total_value: HandValue { soft:  2, hard:  2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " ", bust:  " * * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  1 0 }, Card { suit:  3, value:  1 3 }], total_value: HandValue { soft:  2 3, hard:  2 3 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  4 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  2 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  2 }, Card { suit:  3, value:  1 0 }, Card { suit:  1, value:  4 }, Card { suit:  0, value:  2 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Info { game_id:  0, event: Busts { player: User, score:  2 3 } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  2 3, dealer_score:  1 8 } }
RoundStart { game_id:  0, round_id:  4 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  6 }, Card { suit:  3, value:  6 }], total_value: HandValue { soft:  1 2, hard:  1 2 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  9 }], total_value: HandValue { soft:  9, hard:  9 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * * ", bust:  " * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  3, value:  1 2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " ", bust:  " * * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  6 }, Card { suit:  3, value:  6 }, Card { suit:  3, value:  1 2 }], total_value: HandValue { soft:  2 4, hard:  2 4 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  1 2 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  9 }, Card { suit:  1, value:  7 }, Card { suit:  1, value:  1 2 }], total_value: HandValue { soft:  2 8, hard:  2 8 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 8 } }
Info { game_id:  0, event: Busts { player: User, score:  2 4 } }
GameTied { game_id:  0, user_score:  2 4, dealer_score:  2 8 }
RoundStart { game_id:  0, round_id:  5 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  5 }, Card { suit:  2, value:  1 3 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 3 }], total_value: HandValue { soft:  1 3, hard:  1 3 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * ", bust:  " * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  5 }, Card { suit:  2, value:  1 3 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 3 }, Card { suit:  0, value:  1 0 }], total_value: HandValue { soft:  2 3, hard:  2 3 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 3 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 8, dealer_score:  2 3 } }
RoundStart { game_id:  0, round_id:  6 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  3 }, Card { suit:  3, value:  7 }], total_value: HandValue { soft:  1 0, hard:  1 0 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 2 }], total_value: HandValue { soft:  1 2, hard:  1 2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * * * ", bust:  " * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  2, value:  3 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * ", bust:  " * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  2, value:  3 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * ", bust:  " * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  3 }, Card { suit:  3, value:  7 }, Card { suit:  2, value:  3 }, Card { suit:  2, value:  3 }], total_value: HandValue { soft:  1 6, hard:  1 6 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  3, value:  1 0 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 2 }, Card { suit:  2, value:  2 }, Card { suit:  3, value:  1 0 }], total_value: HandValue { soft:  2 4, hard:  2 4 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 4 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 6, dealer_score:  2 4 } }
RoundStart { game_id:  0, round_id:  7 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  8 }, Card { suit:  3, value:  1 0 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  9 }], total_value: HandValue { soft:  9, hard:  9 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * ", bust:  " * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  8 }, Card { suit:  3, value:  1 0 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  1 2 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  9 }, Card { suit:  1, value:  8 }, Card { suit:  2, value:  1 2 }], total_value: HandValue { soft:  2 9, hard:  2 9 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 9 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 8, dealer_score:  2 9 } }
RoundStart { game_id:  0, round_id:  8 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  8 }, Card { suit:  0, value:  6 }], total_value: HandValue { soft:  1 4, hard:  1 4 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 }], total_value: HandValue { soft:  1, hard:  1 1 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * ", bust:  " * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  2, value:  5 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * ", bust:  " * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  8 }, Card { suit:  0, value:  6 }, Card { suit:  2, value:  5 }], total_value: HandValue { soft:  1 9, hard:  1 9 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  7 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  6 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 }, Card { suit:  1, value:  8 }, Card { suit:  1, value:  7 }, Card { suit:  1, value:  6 }], total_value: HandValue { soft:  2 2, hard:  3 2 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  1 9, dealer_score:  2 2 } }
RoundStart { game_id:  0, round_id:  9 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  1 3 }, Card { suit:  3, value:  1 }], total_value: HandValue { soft:  1 4, hard:  2 4 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  7 }], total_value: HandValue { soft:  7, hard:  7 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * ", bust:  " * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  1, value:  1 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * ", bust:  " * * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  3, value:  4 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * ", bust:  " * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  1 3 }, Card { suit:  3, value:  1 }, Card { suit:  1, value:  1 }, Card { suit:  3, value:  4 }], total_value: HandValue { soft:  1 9, hard:  2 9 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  7 }, Card { suit:  3, value:  1 3 }], total_value: HandValue { soft:  2 0, hard:  2 0 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  1 9, dealer_score:  2 0 } }
RoundStart { game_id:  0, round_id:  1 0 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  7 }, Card { suit:  2, value:  7 }], total_value: HandValue { soft:  1 4, hard:  1 4 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  9 }], total_value: HandValue { soft:  9, hard:  9 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * ", bust:  " * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  1, value:  3 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * ", bust:  " * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  7 }, Card { suit:  2, value:  7 }, Card { suit:  1, value:  3 }], total_value: HandValue { soft:  1 7, hard:  1 7 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  8 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  9 }, Card { suit:  1, value:  1 }, Card { suit:  2, value:  8 }], total_value: HandValue { soft:  1 8, hard:  2 8 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  1 7, dealer_score:  1 8 } }
RoundStart { game_id:  0, round_id:  1 1 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  8 }, Card { suit:  1, value:  1 3 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  2 }], total_value: HandValue { soft:  2, hard:  2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * ", bust:  " * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  8 }, Card { suit:  1, value:  1 3 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  1 3 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  2 }, Card { suit:  2, value:  6 }, Card { suit:  1, value:  1 3 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
GameTied { game_id:  0, user_score:  2 1, dealer_score:  2 1 }
RoundStart { game_id:  0, round_id:  1 2 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  8 }, Card { suit:  3, value:  4 }], total_value: HandValue { soft:  1 2, hard:  1 2 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  3 }], total_value: HandValue { soft:  3, hard:  3 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * * ", bust:  " * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  1, value:  2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * ", bust:  " * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  3, value:  6 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * ", bust:  " * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  8 }, Card { suit:  3, value:  4 }, Card { suit:  1, value:  2 }, Card { suit:  3, value:  6 }], total_value: HandValue { soft:  2 0, hard:  2 0 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  3, value:  8 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  3 }, Card { suit:  3, value:  1 1 }, Card { suit:  3, value:  8 }], total_value: HandValue { soft:  2 2, hard:  2 2 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  2 0, dealer_score:  2 2 } }
RoundStart { game_id:  0, round_id:  1 3 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  4 }, Card { suit:  1, value:  1 1 }], total_value: HandValue { soft:  1 5, hard:  1 5 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  3 }], total_value: HandValue { soft:  3, hard:  3 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * ", bust:  " * * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  0, value:  1 3 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " ", bust:  " * * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  4 }, Card { suit:  1, value:  1 1 }, Card { suit:  0, value:  1 3 }], total_value: HandValue { soft:  2 8, hard:  2 8 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  1 2 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  3 }, Card { suit:  2, value:  1 0 }, Card { suit:  0, value:  1 2 }], total_value: HandValue { soft:  2 5, hard:  2 5 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 5 } }
Info { game_id:  0, event: Busts { player: User, score:  2 8 } }
GameTied { game_id:  0, user_score:  2 8, dealer_score:  2 5 }
RoundStart { game_id:  0, round_id:  1 4 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  1 1 }, Card { suit:  2, value:  1 3 }], total_value: HandValue { soft:  2 4, hard:  2 4 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 2 }], total_value: HandValue { soft:  1 2, hard:  1 2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " ", bust:  " * * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  1 1 }, Card { suit:  2, value:  1 3 }], total_value: HandValue { soft:  2 4, hard:  2 4 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 2 }, Card { suit:  0, value:  1 1 }], total_value: HandValue { soft:  2 3, hard:  2 3 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 3 } }
Info { game_id:  0, event: Busts { player: User, score:  2 4 } }
GameTied { game_id:  0, user_score:  2 4, dealer_score:  2 3 }
RoundStart { game_id:  0, round_id:  1 5 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  9 }, Card { suit:  2, value:  1 }], total_value: HandValue { soft:  1 0, hard:  2 0 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  5 }], total_value: HandValue { soft:  5, hard:  5 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * * * ", bust:  " * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  1, value:  4 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * ", bust:  " * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  3, value:  7 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * ", bust:  " * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  9 }, Card { suit:  2, value:  1 }, Card { suit:  1, value:  4 }, Card { suit:  3, value:  7 }], total_value: HandValue { soft:  2 1, hard:  3 1 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  1 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  1 2 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  5 }, Card { suit:  2, value:  1 0 }, Card { suit:  2, value:  1 }, Card { suit:  0, value:  1 2 }], total_value: HandValue { soft:  2 8, hard:  3 8 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 8 } }
Info { game_id:  0, event: Wins { player: User, user_score:  2 1, dealer_score:  2 8 } }
RoundStart { game_id:  0, round_id:  1 6 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  7 }, Card { suit:  2, value:  1 0 }], total_value: HandValue { soft:  1 7, hard:  1 7 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 0 }], total_value: HandValue { soft:  1 0, hard:  1 0 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * ", bust:  " * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  7 }, Card { suit:  2, value:  1 0 }], total_value: HandValue { soft:  1 7, hard:  1 7 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  3 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  1 3 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 0 }, Card { suit:  0, value:  4 }, Card { suit:  0, value:  3 }, Card { suit:  2, value:  1 3 }], total_value: HandValue { soft:  3 0, hard:  3 0 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  3 0 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 7, dealer_score:  3 0 } }
RoundStart { game_id:  0, round_id:  1 7 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  1 3 }, Card { suit:  1, value:  6 }], total_value: HandValue { soft:  1 9, hard:  1 9 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  5 }], total_value: HandValue { soft:  5, hard:  5 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * ", bust:  " * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  1 3 }, Card { suit:  1, value:  6 }], total_value: HandValue { soft:  1 9, hard:  1 9 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  5 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  3, value:  4 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  5 }, Card { suit:  3, value:  7 }, Card { suit:  0, value:  5 }, Card { suit:  3, value:  4 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  1 9, dealer_score:  2 1 } }
RoundStart { game_id:  0, round_id:  1 8 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  2 }, Card { suit:  0, value:  5 }], total_value: HandValue { soft:  7, hard:  7 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  2 }], total_value: HandValue { soft:  2, hard:  2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * * * * ", bust:  " " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  2, value:  7 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * ", bust:  " * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  1, value:  7 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * ", bust:  " * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  2 }, Card { suit:  0, value:  5 }, Card { suit:  2, value:  7 }, Card { suit:  1, value:  7 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  3, value:  1 1 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  2 }, Card { suit:  1, value:  5 }, Card { suit:  3, value:  1 1 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Info { game_id:  0, event: Wins { player: User, user_score:  2 1, dealer_score:  1 8 } }
RoundStart { game_id:  0, round_id:  1 9 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  1 3 }, Card { suit:  1, value:  4 }], total_value: HandValue { soft:  1 7, hard:  1 7 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  1 }], total_value: HandValue { soft:  1, hard:  1 1 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * ", bust:  " * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  1 3 }, Card { suit:  1, value:  4 }], total_value: HandValue { soft:  1 7, hard:  1 7 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  1 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  5 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  9 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  1 }, Card { suit:  2, value:  3 }, Card { suit:  0, value:  1 }, Card { suit:  2, value:  5 }, Card { suit:  0, value:  9 }], total_value: HandValue { soft:  1 9, hard:  2 9 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  1 7, dealer_score:  1 9 } }
RoundStart { game_id:  0, round_id:  2 0 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  1 0 }, Card { suit:  2, value:  6 }], total_value: HandValue { soft:  1 6, hard:  1 6 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  3 }], total_value: HandValue { soft:  3, hard:  3 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * ", bust:  " * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  1 0 }, Card { suit:  2, value:  6 }], total_value: HandValue { soft:  1 6, hard:  1 6 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  1 3 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  3 }, Card { suit:  3, value:  9 }, Card { suit:  1, value:  1 3 }], total_value: HandValue { soft:  2 5, hard:  2 5 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 5 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 6, dealer_score:  2 5 } }
RoundStart { game_id:  0, round_id:  2 1 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  1 3 }, Card { suit:  2, value:  1 1 }], total_value: HandValue { soft:  2 4, hard:  2 4 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  5 }], total_value: HandValue { soft:  5, hard:  5 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " ", bust:  " * * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  1 3 }, Card { suit:  2, value:  1 1 }], total_value: HandValue { soft:  2 4, hard:  2 4 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  3, value:  5 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  1 2 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  5 }, Card { suit:  2, value:  1 }, Card { suit:  3, value:  5 }, Card { suit:  1, value:  1 2 }], total_value: HandValue { soft:  2 3, hard:  3 3 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 3 } }
Info { game_id:  0, event: Busts { player: User, score:  2 4 } }
GameTied { game_id:  0, user_score:  2 4, dealer_score:  2 3 }
RoundStart { game_id:  0, round_id:  2 2 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  9 }, Card { suit:  1, value:  1 1 }], total_value: HandValue { soft:  2 0, hard:  2 0 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  2 }], total_value: HandValue { soft:  2, hard:  2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * ", bust:  " * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  9 }, Card { suit:  1, value:  1 1 }], total_value: HandValue { soft:  2 0, hard:  2 0 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  1 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  6 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  3 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  2 }, Card { suit:  3, value:  8 }, Card { suit:  1, value:  1 }, Card { suit:  0, value:  6 }, Card { suit:  1, value:  3 }], total_value: HandValue { soft:  2 0, hard:  3 0 } } }
GameTied { game_id:  0, user_score:  2 0, dealer_score:  2 0 }
RoundStart { game_id:  0, round_id:  2 3 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  3 }, Card { suit:  3, value:  1 3 }], total_value: HandValue { soft:  1 6, hard:  1 6 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  6 }], total_value: HandValue { soft:  6, hard:  6 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * ", bust:  " * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  3 }, Card { suit:  3, value:  1 3 }], total_value: HandValue { soft:  1 6, hard:  1 6 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  1 1 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  6 }, Card { suit:  0, value:  5 }, Card { suit:  2, value:  1 1 }], total_value: HandValue { soft:  2 2, hard:  2 2 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  1 6, dealer_score:  2 2 } }
RoundStart { game_id:  0, round_id:  2 4 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  8 }, Card { suit:  0, value:  6 }], total_value: HandValue { soft:  1 4, hard:  1 4 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  4 }], total_value: HandValue { soft:  4, hard:  4 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * ", bust:  " * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  1, value:  6 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * ", bust:  " * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  8 }, Card { suit:  0, value:  6 }, Card { suit:  1, value:  6 }], total_value: HandValue { soft:  2 0, hard:  2 0 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  2 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  3, value:  2 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  1 2 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  4 }, Card { suit:  1, value:  9 }, Card { suit:  0, value:  2 }, Card { suit:  3, value:  2 }, Card { suit:  1, value:  1 2 }], total_value: HandValue { soft:  2 9, hard:  2 9 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 9 } }
Info { game_id:  0, event: Wins { player: User, user_score:  2 0, dealer_score:  2 9 } }
RoundStart { game_id:  0, round_id:  2 5 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  3 }, Card { suit:  1, value:  1 1 }], total_value: HandValue { soft:  1 4, hard:  1 4 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  1 2 }], total_value: HandValue { soft:  1 2, hard:  1 2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * ", bust:  " * * * * " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  3, value:  4 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * ", bust:  " * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  3 }, Card { suit:  1, value:  1 1 }, Card { suit:  3, value:  4 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  2, value:  1 2 }, Card { suit:  2, value:  8 }], total_value: HandValue { soft:  2 0, hard:  2 0 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  1 8, dealer_score:  2 0 } }
RoundStart { game_id:  0, round_id:  2 6 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  5 }, Card { suit:  2, value:  2 }], total_value: HandValue { soft:  7, hard:  7 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  1 }], total_value: HandValue { soft:  1, hard:  1 1 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * * * * ", bust:  " " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  3, value:  8 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * ", bust:  " * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  5 }, Card { suit:  2, value:  2 }, Card { suit:  3, value:  8 }], total_value: HandValue { soft:  1 5, hard:  1 5 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  8 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  1 }, Card { suit:  1, value:  1 0 }, Card { suit:  0, value:  8 }], total_value: HandValue { soft:  1 9, hard:  2 9 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  1 5, dealer_score:  1 9 } }
RoundStart { game_id:  0, round_id:  2 7 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  1 0 }, Card { suit:  0, value:  9 }], total_value: HandValue { soft:  1 9, hard:  1 9 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  1 }], total_value: HandValue { soft:  1, hard:  1 1 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * ", bust:  " * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  1 0 }, Card { suit:  0, value:  9 }], total_value: HandValue { soft:  1 9, hard:  1 9 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  4 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  4 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  2 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  3, value:  1 1 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  1 }, Card { suit:  2, value:  4 }, Card { suit:  0, value:  4 }, Card { suit:  0, value:  4 }, Card { suit:  1, value:  2 }, Card { suit:  3, value:  1 1 }], total_value: HandValue { soft:  2 6, hard:  3 6 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 6 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 9, dealer_score:  2 6 } }
RoundStart { game_id:  0, round_id:  2 8 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  1 1 }, Card { suit:  3, value:  7 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  2 }], total_value: HandValue { soft:  2, hard:  2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * ", bust:  " * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  1 1 }, Card { suit:  3, value:  7 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  3, value:  6 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  8 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  2 }, Card { suit:  2, value:  4 }, Card { suit:  3, value:  6 }, Card { suit:  0, value:  8 }], total_value: HandValue { soft:  2 0, hard:  2 0 } } }
Info { game_id:  0, event: Wins { player: Dealer, user_score:  1 8, dealer_score:  2 0 } }
RoundStart { game_id:  0, round_id:  2 9 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  9 }, Card { suit:  3, value:  1 2 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  2 }], total_value: HandValue { soft:  2, hard:  2 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * ", bust:  " * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  9 }, Card { suit:  3, value:  1 2 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  1, value:  5 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  8 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  2 }, Card { suit:  0, value:  1 0 }, Card { suit:  1, value:  5 }, Card { suit:  2, value:  8 }], total_value: HandValue { soft:  2 5, hard:  2 5 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 5 } }
Info { game_id:  0, event: Wins { player: User, user_score:  2 1, dealer_score:  2 5 } }
RoundStart { game_id:  0, round_id:  3 0 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  9 }, Card { suit:  2, value:  9 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  1 }], total_value: HandValue { soft:  1, hard:  1 1 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * ", bust:  " * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  0, value:  9 }, Card { suit:  2, value:  9 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  1 0 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  3, value:  1 }, Card { suit:  3, value:  1 2 }, Card { suit:  2, value:  1 0 }], total_value: HandValue { soft:  2 3, hard:  3 3 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 3 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 8, dealer_score:  2 3 } }
RoundStart { game_id:  0, round_id:  3 1 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  3 }, Card { suit:  2, value:  7 }], total_value: HandValue { soft:  1 0, hard:  1 0 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  6 }], total_value: HandValue { soft:  6, hard:  6 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * * * * ", bust:  " " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  1, value:  1 1 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * ", bust:  " * * * * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  3, value:  3 }, Card { suit:  2, value:  7 }, Card { suit:  1, value:  1 1 }], total_value: HandValue { soft:  2 1, hard:  2 1 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  1, value:  6 }, Card { suit:  2, value:  1 2 }], total_value: HandValue { soft:  1 8, hard:  1 8 } } }
Info { game_id:  0, event: Wins { player: User, user_score:  2 1, dealer_score:  1 8 } }
RoundStart { game_id:  0, round_id:  3 2 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  7 }, Card { suit:  3, value:  9 }], total_value: HandValue { soft:  1 6, hard:  1 6 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 1 }], total_value: HandValue { soft:  1 1, hard:  1 1 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * ", bust:  " * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  1, value:  7 }, Card { suit:  3, value:  9 }], total_value: HandValue { soft:  1 6, hard:  1 6 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  2, value:  1 1 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  1 1 }, Card { suit:  0, value:  3 }, Card { suit:  2, value:  1 1 }], total_value: HandValue { soft:  2 5, hard:  2 5 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 5 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 6, dealer_score:  2 5 } }
RoundStart { game_id:  0, round_id:  3 3 }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  6 }, Card { suit:  2, value:  2 }], total_value: HandValue { soft:  8, hard:  8 } } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  4 }], total_value: HandValue { soft:  4, hard:  4 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * * * * * * * ", bust:  " " } }
Action { game_id:  0, event: Hits { player: User, card: Card { suit:  1, value:  8 } } }
Info { game_id:  0, event: Probabilities { player: User, improvement:  " * * * * ", bust:  " * * * * * * " } }
Action { game_id:  0, event: Stands { player: User } }
Info { game_id:  0, event: Hand { player: User, cards: [Card { suit:  2, value:  6 }, Card { suit:  2, value:  2 }, Card { suit:  1, value:  8 }], total_value: HandValue { soft:  1 6, hard:  1 6 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  1 } } }
Action { game_id:  0, event: Hits { player: Dealer, card: Card { suit:  0, value:  1 1 } } }
Action { game_id:  0, event: Stands { player: Dealer } }
Info { game_id:  0, event: Hand { player: Dealer, cards: [Card { suit:  0, value:  4 }, Card { suit:  1, value:  1 2 }, Card { suit:  0, value:  1 }, Card { suit:  0, value:  1 1 }], total_value: HandValue { soft:  2 8, hard:  3 8 } } }
Info { game_id:  0, event: Busts { player: Dealer, score:  2 8 } }
Info { game_id:  0, event: Wins { player: User, user_score:  1 6, dealer_score:  2 8 } }
```
