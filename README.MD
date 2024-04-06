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

```
GameStart { game_id: 0, num_suits: 4, num_cards_per_suit: 13, num_decks: 4, target_score: 22, dealer_threshold: 18 }
RoundStart { game_id: 0, round_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 11 }, Card { suit: 2, value: 7 }], total_value: HandValue { soft: 18, hard: 18 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 7 }], total_value: HandValue { soft: 7, hard: 7 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "***", bust: "*******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 11 }, Card { suit: 2, value: 7 }], total_value: HandValue { soft: 18, hard: 18 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 12 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 7 }, Card { suit: 1, value: 9 }, Card { suit: 0, value: 12 }], total_value: HandValue { soft: 28, hard: 28 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 28 } }
Info { game_id: 0, event: Wins { player: User, user_score: 18, dealer_score: 28 } }
RoundStart { game_id: 0, round_id: 1 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 5 }, Card { suit: 1, value: 3 }], total_value: HandValue { soft: 8, hard: 8 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 7 }], total_value: HandValue { soft: 7, hard: 7 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 3, value: 12 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**", bust: "********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 5 }, Card { suit: 1, value: 3 }, Card { suit: 3, value: 12 }], total_value: HandValue { soft: 20, hard: 20 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 7 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 8 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 7 }, Card { suit: 1, value: 2 }, Card { suit: 3, value: 7 }, Card { suit: 3, value: 8 }], total_value: HandValue { soft: 24, hard: 24 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 24 } }
Info { game_id: 0, event: Wins { player: User, user_score: 20, dealer_score: 24 } }
RoundStart { game_id: 0, round_id: 2 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 9 }, Card { suit: 2, value: 1 }], total_value: HandValue { soft: 10, hard: 20 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 1 }], total_value: HandValue { soft: 1, hard: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*********", bust: "*" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 3, value: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*", bust: "*********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 9 }, Card { suit: 2, value: 1 }, Card { suit: 3, value: 11 }], total_value: HandValue { soft: 21, hard: 31 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 8 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 1, value: 13 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 1 }, Card { suit: 0, value: 5 }, Card { suit: 0, value: 8 }, Card { suit: 1, value: 13 }], total_value: HandValue { soft: 27, hard: 37 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 27 } }
Info { game_id: 0, event: Wins { player: User, user_score: 21, dealer_score: 27 } }
RoundStart { game_id: 0, round_id: 3 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 9 }, Card { suit: 1, value: 11 }], total_value: HandValue { soft: 20, hard: 20 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 13 }], total_value: HandValue { soft: 13, hard: 13 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**", bust: "********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 9 }, Card { suit: 1, value: 11 }], total_value: HandValue { soft: 20, hard: 20 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 13 }, Card { suit: 2, value: 13 }], total_value: HandValue { soft: 26, hard: 26 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 26 } }
Info { game_id: 0, event: Wins { player: User, user_score: 20, dealer_score: 26 } }
RoundStart { game_id: 0, round_id: 4 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 4 }, Card { suit: 1, value: 13 }], total_value: HandValue { soft: 17, hard: 17 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 5 }], total_value: HandValue { soft: 5, hard: 5 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "****", bust: "******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 4 }, Card { suit: 1, value: 13 }], total_value: HandValue { soft: 17, hard: 17 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 9 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 5 }, Card { suit: 1, value: 5 }, Card { suit: 2, value: 9 }], total_value: HandValue { soft: 19, hard: 19 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 17, dealer_score: 19 } }
RoundStart { game_id: 0, round_id: 5 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 8 }, Card { suit: 2, value: 7 }], total_value: HandValue { soft: 15, hard: 15 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 4 }], total_value: HandValue { soft: 4, hard: 4 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*****", bust: "*****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 2, value: 13 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 8 }, Card { suit: 2, value: 7 }, Card { suit: 2, value: 13 }], total_value: HandValue { soft: 28, hard: 28 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 13 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 4 }, Card { suit: 2, value: 3 }, Card { suit: 3, value: 13 }], total_value: HandValue { soft: 20, hard: 20 } } }
Info { game_id: 0, event: Busts { player: User, score: 28 } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 28, dealer_score: 20 } }
RoundStart { game_id: 0, round_id: 6 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 7 }, Card { suit: 1, value: 7 }], total_value: HandValue { soft: 14, hard: 14 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 12 }], total_value: HandValue { soft: 12, hard: 12 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 3, value: 10 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 7 }, Card { suit: 1, value: 7 }, Card { suit: 3, value: 10 }], total_value: HandValue { soft: 24, hard: 24 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 13 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 12 }, Card { suit: 3, value: 2 }, Card { suit: 3, value: 13 }], total_value: HandValue { soft: 27, hard: 27 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 27 } }
Info { game_id: 0, event: Busts { player: User, score: 24 } }
GameTied { game_id: 0, user_score: 24, dealer_score: 27 }
RoundStart { game_id: 0, round_id: 7 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 11 }, Card { suit: 1, value: 9 }], total_value: HandValue { soft: 20, hard: 20 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 9 }], total_value: HandValue { soft: 9, hard: 9 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**", bust: "********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 11 }, Card { suit: 1, value: 9 }], total_value: HandValue { soft: 20, hard: 20 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 9 }, Card { suit: 3, value: 12 }], total_value: HandValue { soft: 21, hard: 21 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 20, dealer_score: 21 } }
RoundStart { game_id: 0, round_id: 8 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 7 }, Card { suit: 0, value: 8 }], total_value: HandValue { soft: 15, hard: 15 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 10 }], total_value: HandValue { soft: 10, hard: 10 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 3 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "***", bust: "*******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 7 }, Card { suit: 0, value: 8 }, Card { suit: 0, value: 3 }], total_value: HandValue { soft: 18, hard: 18 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 1, value: 8 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 10 }, Card { suit: 1, value: 6 }, Card { suit: 1, value: 8 }], total_value: HandValue { soft: 24, hard: 24 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 24 } }
Info { game_id: 0, event: Wins { player: User, user_score: 18, dealer_score: 24 } }
RoundStart { game_id: 0, round_id: 9 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 10 }, Card { suit: 2, value: 4 }], total_value: HandValue { soft: 14, hard: 14 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 5 }], total_value: HandValue { soft: 5, hard: 5 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 1, value: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 10 }, Card { suit: 2, value: 4 }, Card { suit: 1, value: 11 }], total_value: HandValue { soft: 25, hard: 25 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 1 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 7 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 1, value: 11 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 5 }, Card { suit: 0, value: 2 }, Card { suit: 0, value: 1 }, Card { suit: 2, value: 7 }, Card { suit: 1, value: 11 }], total_value: HandValue { soft: 26, hard: 36 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 26 } }
Info { game_id: 0, event: Busts { player: User, score: 25 } }
GameTied { game_id: 0, user_score: 25, dealer_score: 26 }
RoundStart { game_id: 0, round_id: 10 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 4 }, Card { suit: 3, value: 12 }], total_value: HandValue { soft: 16, hard: 16 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 9 }], total_value: HandValue { soft: 9, hard: 9 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*****", bust: "*****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 9 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 4 }, Card { suit: 3, value: 12 }, Card { suit: 0, value: 9 }], total_value: HandValue { soft: 25, hard: 25 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 9 }, Card { suit: 2, value: 9 }], total_value: HandValue { soft: 18, hard: 18 } } }
Info { game_id: 0, event: Busts { player: User, score: 25 } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 25, dealer_score: 18 } }
RoundStart { game_id: 0, round_id: 11 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 9 }, Card { suit: 1, value: 12 }], total_value: HandValue { soft: 21, hard: 21 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 4 }], total_value: HandValue { soft: 4, hard: 4 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*", bust: "*********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 9 }, Card { suit: 1, value: 12 }], total_value: HandValue { soft: 21, hard: 21 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 1, value: 9 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 4 }, Card { suit: 3, value: 5 }, Card { suit: 1, value: 9 }], total_value: HandValue { soft: 18, hard: 18 } } }
Info { game_id: 0, event: Wins { player: User, user_score: 21, dealer_score: 18 } }
RoundStart { game_id: 0, round_id: 12 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 4 }, Card { suit: 2, value: 1 }], total_value: HandValue { soft: 5, hard: 15 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 12 }], total_value: HandValue { soft: 12, hard: 12 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 3, value: 13 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "****", bust: "******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 4 }, Card { suit: 2, value: 1 }, Card { suit: 3, value: 13 }], total_value: HandValue { soft: 18, hard: 28 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 12 }, Card { suit: 0, value: 6 }], total_value: HandValue { soft: 18, hard: 18 } } }
GameTied { game_id: 0, user_score: 18, dealer_score: 18 }
RoundStart { game_id: 0, round_id: 13 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 3 }, Card { suit: 0, value: 10 }], total_value: HandValue { soft: 13, hard: 13 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 12 }], total_value: HandValue { soft: 12, hard: 12 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 2, value: 6 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "***", bust: "*******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 3 }, Card { suit: 0, value: 10 }, Card { suit: 2, value: 6 }], total_value: HandValue { soft: 19, hard: 19 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 6 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 12 }, Card { suit: 3, value: 5 }, Card { suit: 0, value: 6 }], total_value: HandValue { soft: 23, hard: 23 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 23 } }
Info { game_id: 0, event: Wins { player: User, user_score: 19, dealer_score: 23 } }
RoundStart { game_id: 0, round_id: 14 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 7 }, Card { suit: 1, value: 8 }], total_value: HandValue { soft: 15, hard: 15 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 13 }], total_value: HandValue { soft: 13, hard: 13 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 2, value: 12 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 7 }, Card { suit: 1, value: 8 }, Card { suit: 2, value: 12 }], total_value: HandValue { soft: 27, hard: 27 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 13 }, Card { suit: 3, value: 9 }], total_value: HandValue { soft: 22, hard: 22 } } }
Info { game_id: 0, event: Busts { player: User, score: 27 } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 27, dealer_score: 22 } }
RoundStart { game_id: 0, round_id: 15 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 10 }, Card { suit: 2, value: 5 }], total_value: HandValue { soft: 15, hard: 15 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 8 }], total_value: HandValue { soft: 8, hard: 8 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 10 }, Card { suit: 2, value: 5 }, Card { suit: 0, value: 11 }], total_value: HandValue { soft: 26, hard: 26 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 1, value: 13 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 8 }, Card { suit: 1, value: 8 }, Card { suit: 1, value: 13 }], total_value: HandValue { soft: 29, hard: 29 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 29 } }
Info { game_id: 0, event: Busts { player: User, score: 26 } }
GameTied { game_id: 0, user_score: 26, dealer_score: 29 }
RoundStart { game_id: 0, round_id: 16 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 2 }, Card { suit: 1, value: 1 }], total_value: HandValue { soft: 3, hard: 13 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 6 }], total_value: HandValue { soft: 6, hard: 6 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 2, value: 10 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 2 }, Card { suit: 1, value: 1 }, Card { suit: 0, value: 11 }, Card { suit: 2, value: 10 }], total_value: HandValue { soft: 24, hard: 34 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 4 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 6 }, Card { suit: 0, value: 9 }, Card { suit: 0, value: 4 }], total_value: HandValue { soft: 19, hard: 19 } } }
Info { game_id: 0, event: Busts { player: User, score: 24 } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 24, dealer_score: 19 } }
RoundStart { game_id: 0, round_id: 17 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 10 }, Card { suit: 0, value: 4 }], total_value: HandValue { soft: 14, hard: 14 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 13 }], total_value: HandValue { soft: 13, hard: 13 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 10 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 10 }, Card { suit: 0, value: 4 }, Card { suit: 0, value: 10 }], total_value: HandValue { soft: 24, hard: 24 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 10 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 13 }, Card { suit: 3, value: 2 }, Card { suit: 0, value: 10 }], total_value: HandValue { soft: 25, hard: 25 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 25 } }
Info { game_id: 0, event: Busts { player: User, score: 24 } }
GameTied { game_id: 0, user_score: 24, dealer_score: 25 }
RoundStart { game_id: 0, round_id: 18 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 3 }, Card { suit: 3, value: 11 }], total_value: HandValue { soft: 14, hard: 14 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 8 }], total_value: HandValue { soft: 8, hard: 8 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 1, value: 3 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*****", bust: "*****" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 3 }, Card { suit: 3, value: 11 }, Card { suit: 1, value: 3 }], total_value: HandValue { soft: 17, hard: 17 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 8 }, Card { suit: 0, value: 13 }], total_value: HandValue { soft: 21, hard: 21 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 17, dealer_score: 21 } }
RoundStart { game_id: 0, round_id: 19 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 6 }, Card { suit: 2, value: 6 }], total_value: HandValue { soft: 12, hard: 12 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 2 }], total_value: HandValue { soft: 2, hard: 2 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "********", bust: "**" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 1, value: 6 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "****", bust: "******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 6 }, Card { suit: 2, value: 6 }, Card { suit: 1, value: 6 }], total_value: HandValue { soft: 18, hard: 18 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 5 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 5 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 2 }, Card { suit: 0, value: 7 }, Card { suit: 2, value: 5 }, Card { suit: 2, value: 5 }], total_value: HandValue { soft: 19, hard: 19 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 18, dealer_score: 19 } }
RoundStart { game_id: 0, round_id: 20 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 7 }, Card { suit: 0, value: 1 }], total_value: HandValue { soft: 8, hard: 18 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 7 }], total_value: HandValue { soft: 7, hard: 7 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 2, value: 5 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 8 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*", bust: "*********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 7 }, Card { suit: 0, value: 1 }, Card { suit: 2, value: 5 }, Card { suit: 0, value: 8 }], total_value: HandValue { soft: 21, hard: 31 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 4 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 1 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 7 }, Card { suit: 0, value: 6 }, Card { suit: 3, value: 4 }, Card { suit: 3, value: 1 }], total_value: HandValue { soft: 18, hard: 28 } } }
Info { game_id: 0, event: Wins { player: User, user_score: 21, dealer_score: 18 } }
RoundStart { game_id: 0, round_id: 21 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 5 }, Card { suit: 2, value: 11 }], total_value: HandValue { soft: 16, hard: 16 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 1 }], total_value: HandValue { soft: 1, hard: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 2, value: 12 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 5 }, Card { suit: 2, value: 11 }, Card { suit: 2, value: 12 }], total_value: HandValue { soft: 28, hard: 28 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 1, value: 10 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 1 }, Card { suit: 3, value: 10 }, Card { suit: 1, value: 10 }], total_value: HandValue { soft: 21, hard: 31 } } }
Info { game_id: 0, event: Busts { player: User, score: 28 } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 28, dealer_score: 21 } }
RoundStart { game_id: 0, round_id: 22 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 5 }, Card { suit: 1, value: 1 }], total_value: HandValue { soft: 6, hard: 16 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 1 }], total_value: HandValue { soft: 1, hard: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 8 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 1 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 1, value: 4 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "***", bust: "*******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 5 }, Card { suit: 1, value: 1 }, Card { suit: 0, value: 8 }, Card { suit: 0, value: 1 }, Card { suit: 1, value: 4 }], total_value: HandValue { soft: 19, hard: 29 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 11 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 2 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 1 }, Card { suit: 3, value: 4 }, Card { suit: 3, value: 11 }, Card { suit: 2, value: 2 }], total_value: HandValue { soft: 18, hard: 28 } } }
Info { game_id: 0, event: Wins { player: User, user_score: 19, dealer_score: 18 } }
RoundStart { game_id: 0, round_id: 23 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 2 }, Card { suit: 3, value: 6 }], total_value: HandValue { soft: 8, hard: 8 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 4 }], total_value: HandValue { soft: 4, hard: 4 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 3, value: 6 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 5 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "***", bust: "*******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 2 }, Card { suit: 3, value: 6 }, Card { suit: 3, value: 6 }, Card { suit: 0, value: 5 }], total_value: HandValue { soft: 19, hard: 19 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 7 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 3 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 10 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 4 }, Card { suit: 2, value: 2 }, Card { suit: 3, value: 7 }, Card { suit: 0, value: 3 }, Card { suit: 3, value: 10 }], total_value: HandValue { soft: 26, hard: 26 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 26 } }
Info { game_id: 0, event: Wins { player: User, user_score: 19, dealer_score: 26 } }
RoundStart { game_id: 0, round_id: 24 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 3 }, Card { suit: 1, value: 10 }], total_value: HandValue { soft: 13, hard: 13 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 2 }], total_value: HandValue { soft: 2, hard: 2 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 2 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 3 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "****", bust: "******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 3 }, Card { suit: 1, value: 10 }, Card { suit: 0, value: 2 }, Card { suit: 0, value: 3 }], total_value: HandValue { soft: 18, hard: 18 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 3 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 10 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 2 }, Card { suit: 3, value: 6 }, Card { suit: 2, value: 3 }, Card { suit: 2, value: 10 }], total_value: HandValue { soft: 21, hard: 21 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 18, dealer_score: 21 } }
RoundStart { game_id: 0, round_id: 25 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 7 }, Card { suit: 0, value: 3 }], total_value: HandValue { soft: 10, hard: 10 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 2 }], total_value: HandValue { soft: 2, hard: 2 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*********", bust: "*" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 10 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**", bust: "********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 7 }, Card { suit: 0, value: 3 }, Card { suit: 0, value: 10 }], total_value: HandValue { soft: 20, hard: 20 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 5 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 1 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 2 }, Card { suit: 2, value: 4 }, Card { suit: 0, value: 5 }, Card { suit: 3, value: 1 }], total_value: HandValue { soft: 12, hard: 22 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 20, dealer_score: 22 } }
RoundStart { game_id: 0, round_id: 26 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 2 }, Card { suit: 1, value: 2 }], total_value: HandValue { soft: 4, hard: 4 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 9 }], total_value: HandValue { soft: 9, hard: 9 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 1, value: 9 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 2, value: 3 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*****", bust: "*****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 2, value: 3 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**", bust: "********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 2 }, Card { suit: 1, value: 2 }, Card { suit: 1, value: 9 }, Card { suit: 2, value: 3 }, Card { suit: 2, value: 3 }], total_value: HandValue { soft: 19, hard: 19 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 3 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 6 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 9 }, Card { suit: 2, value: 1 }, Card { suit: 3, value: 3 }, Card { suit: 2, value: 6 }], total_value: HandValue { soft: 19, hard: 29 } } }
GameTied { game_id: 0, user_score: 19, dealer_score: 19 }
RoundStart { game_id: 0, round_id: 27 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 5 }, Card { suit: 0, value: 11 }], total_value: HandValue { soft: 16, hard: 16 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 1 }], total_value: HandValue { soft: 1, hard: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*****", bust: "*****" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 5 }, Card { suit: 0, value: 11 }], total_value: HandValue { soft: 16, hard: 16 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 12 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 10 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 1 }, Card { suit: 0, value: 4 }, Card { suit: 0, value: 12 }, Card { suit: 2, value: 10 }], total_value: HandValue { soft: 27, hard: 37 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 27 } }
Info { game_id: 0, event: Wins { player: User, user_score: 16, dealer_score: 27 } }
RoundStart { game_id: 0, round_id: 28 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 12 }, Card { suit: 1, value: 4 }], total_value: HandValue { soft: 16, hard: 16 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 6 }], total_value: HandValue { soft: 6, hard: 6 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*****", bust: "*****" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 12 }, Card { suit: 1, value: 4 }], total_value: HandValue { soft: 16, hard: 16 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 2 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 0, value: 2 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 1 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 1, value: 3 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 6 }, Card { suit: 2, value: 4 }, Card { suit: 3, value: 2 }, Card { suit: 0, value: 2 }, Card { suit: 3, value: 1 }, Card { suit: 1, value: 3 }], total_value: HandValue { soft: 18, hard: 28 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 16, dealer_score: 18 } }
RoundStart { game_id: 0, round_id: 29 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 6 }, Card { suit: 1, value: 3 }], total_value: HandValue { soft: 9, hard: 9 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 11 }], total_value: HandValue { soft: 11, hard: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 3, value: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*", bust: "*********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 6 }, Card { suit: 1, value: 3 }, Card { suit: 3, value: 11 }], total_value: HandValue { soft: 20, hard: 20 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 11 }, Card { suit: 2, value: 13 }], total_value: HandValue { soft: 24, hard: 24 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 24 } }
Info { game_id: 0, event: Wins { player: User, user_score: 20, dealer_score: 24 } }
RoundStart { game_id: 0, round_id: 30 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 12 }, Card { suit: 0, value: 13 }], total_value: HandValue { soft: 25, hard: 25 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 8 }], total_value: HandValue { soft: 8, hard: 8 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 12 }, Card { suit: 0, value: 13 }], total_value: HandValue { soft: 25, hard: 25 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 8 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 8 }, Card { suit: 3, value: 8 }, Card { suit: 2, value: 8 }], total_value: HandValue { soft: 24, hard: 24 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 24 } }
Info { game_id: 0, event: Busts { player: User, score: 25 } }
GameTied { game_id: 0, user_score: 25, dealer_score: 24 }
RoundStart { game_id: 0, round_id: 31 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 11 }, Card { suit: 2, value: 13 }], total_value: HandValue { soft: 24, hard: 24 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 12 }], total_value: HandValue { soft: 12, hard: 12 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 11 }, Card { suit: 2, value: 13 }], total_value: HandValue { soft: 24, hard: 24 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 12 }, Card { suit: 3, value: 8 }], total_value: HandValue { soft: 20, hard: 20 } } }
Info { game_id: 0, event: Busts { player: User, score: 24 } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 24, dealer_score: 20 } }
RoundStart { game_id: 0, round_id: 32 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 13 }, Card { suit: 0, value: 11 }], total_value: HandValue { soft: 24, hard: 24 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 8 }], total_value: HandValue { soft: 8, hard: 8 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 3, value: 13 }, Card { suit: 0, value: 11 }], total_value: HandValue { soft: 24, hard: 24 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 12 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 8 }, Card { suit: 0, value: 7 }, Card { suit: 2, value: 12 }], total_value: HandValue { soft: 27, hard: 27 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 27 } }
Info { game_id: 0, event: Busts { player: User, score: 24 } }
GameTied { game_id: 0, user_score: 24, dealer_score: 27 }
```
