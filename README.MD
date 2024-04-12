## Usage

<!-- cargo run --release -- --help | pbcopy -->
```
Usage: generate-data [--min-decks=ARG] [--max-decks=ARG] [--min-suits=ARG] [--max-suits=ARG] [
--min-cards-per-suit=ARG] [--max-cards-per-suit=ARG] [-n=ARG] [-m=ARG] [--output=ARG]

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
        --output=ARG     Prefix for output files (default: blackjack_transcripts)
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

<!-- rm blackjack_transcripts.txt && cargo run --release -- -n 60 -m 2 --min-decks=1 --max-decks=1 --min-suits=2 --max-suits=4 --min-cards-per-suit=6 --max-cards-per-suit=15 | pbcopy -->
```
User wins:      10,921 (41.86%)
Dealer wins:    10,575 (40.53%)
Ties:           4,596 (17.61%)
Total rounds:   26,092
Total games:    5,400
Total sessions: 3,600
```

## Example

```sh
cargo run --release -- \
    -n 1 -m 1 \
    --min-decks=1 \
    --max-decks=1 \
    --min-suits=4 \
    --max-suits=4 \
    --min-cards-per-suit=13 \
    --max-cards-per-suit=13
```

```
User wins:      2 (22.22%)
Dealer wins:    6 (66.67%)
Ties:           1 (11.11%)
Total rounds:   9
Total games:    1
Total sessions: 1
Creating tokenizer...
[00:00:00] Pre-processing files (0 Mo)    ████████████████████████████████████████████████████████████████████████████████████████                100%Tokenizing...
Number of tokens:               2,774
Tokens encoded per second:      1452387.20
Average tokens per session:     2774.00
Longest session (tokens):       2,774
Decoding...
```

## Target Score and Dealer Threshold

On average, we want at least one useful hit or stand decision per game, so -

```rs
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
```

| Suits | Cards per Suit | Target Score | Dealer Threshold |
|-------|----------------|--------------|------------------|
| 2 | 6 | 16 | 13 |
| 2 | 7 | 16 | 14 |
| 2 | 8 | 17 | 14 |
| 2 | 9 | 18 | 15 |
| 2 | 10 | 20 | 16 |
| 2 | 11 | 20 | 17 |
| 2 | 12 | 21 | 18 |
| 2 | 13 | 22 | 18 |
| 2 | 14 | 23 | 19 |
| 2 | 15 | 23 | 19 |
| 3 | 6 | 16 | 13 |
| 3 | 7 | 16 | 14 |
| 3 | 8 | 17 | 14 |
| 3 | 9 | 18 | 15 |
| 3 | 10 | 20 | 16 |
| 3 | 11 | 20 | 17 |
| 3 | 12 | 21 | 18 |
| 3 | 13 | 22 | 18 |
| 3 | 14 | 23 | 19 |
| 3 | 15 | 23 | 19 |
| 4 | 6 | 16 | 13 |
| 4 | 7 | 16 | 14 |
| 4 | 8 | 17 | 14 |
| 4 | 9 | 18 | 15 |
| 4 | 10 | 20 | 16 |
| 4 | 11 | 20 | 17 |
| 4 | 12 | 21 | 18 |
| 4 | 13 | 22 | 18 |
| 4 | 14 | 23 | 19 |
| 4 | 15 | 23 | 19 |

## Generated Data

<details>
<summary>Click to expand</summary>

```
BeginSession
GameStart { game_id: 0, num_suits: 4, num_cards_per_suit: 13, num_decks: 1, target_score: 22, dealer_threshold: 18 }
RoundStart { game_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 7 }, Card { suit: 3, value: 4 }], total_value: HandValue { soft: 11, hard: 11 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 3 }], total_value: HandValue { soft: 3, hard: 3 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "********", bust: "**" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 1, value: 7 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "***", bust: "*******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 7 }, Card { suit: 3, value: 4 }, Card { suit: 1, value: 7 }], total_value: HandValue { soft: 18, hard: 18 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 2, value: 7 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 3 }, Card { suit: 3, value: 12 }, Card { suit: 2, value: 7 }], total_value: HandValue { soft: 22, hard: 22 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 18, dealer_score: 22 } }
RoundEnd { game_id: 0 }
RoundStart { game_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 11 }, Card { suit: 2, value: 2 }], total_value: HandValue { soft: 13, hard: 13 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 2 }], total_value: HandValue { soft: 2, hard: 2 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 1, value: 4 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "***", bust: "*******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 11 }, Card { suit: 2, value: 2 }, Card { suit: 1, value: 4 }], total_value: HandValue { soft: 17, hard: 17 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 10 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 3, value: 2 }, Card { suit: 2, value: 11 }, Card { suit: 3, value: 10 }], total_value: HandValue { soft: 23, hard: 23 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 23 } }
Info { game_id: 0, event: Wins { player: User, user_score: 17, dealer_score: 23 } }
RoundEnd { game_id: 0 }
RoundStart { game_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 10 }, Card { suit: 0, value: 4 }], total_value: HandValue { soft: 14, hard: 14 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 5 }], total_value: HandValue { soft: 5, hard: 5 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 12 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 10 }, Card { suit: 0, value: 4 }, Card { suit: 0, value: 12 }], total_value: HandValue { soft: 26, hard: 26 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 5 }, Card { suit: 1, value: 13 }], total_value: HandValue { soft: 18, hard: 18 } } }
Info { game_id: 0, event: Busts { player: User, score: 26 } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 26, dealer_score: 18 } }
RoundEnd { game_id: 0 }
RoundStart { game_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 9 }, Card { suit: 3, value: 5 }], total_value: HandValue { soft: 14, hard: 14 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 8 }], total_value: HandValue { soft: 8, hard: 8 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "******", bust: "****" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 6 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**", bust: "********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 9 }, Card { suit: 3, value: 5 }, Card { suit: 0, value: 6 }], total_value: HandValue { soft: 20, hard: 20 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 8 }, Card { suit: 0, value: 13 }], total_value: HandValue { soft: 21, hard: 21 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 20, dealer_score: 21 } }
RoundEnd { game_id: 0 }
RoundStart { game_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 5 }, Card { suit: 3, value: 7 }], total_value: HandValue { soft: 12, hard: 12 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 9 }], total_value: HandValue { soft: 9, hard: 9 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*******", bust: "***" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 2, value: 3 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "****", bust: "******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 5 }, Card { suit: 3, value: 7 }, Card { suit: 2, value: 3 }], total_value: HandValue { soft: 15, hard: 15 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 9 }, Card { suit: 2, value: 10 }], total_value: HandValue { soft: 19, hard: 19 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 15, dealer_score: 19 } }
RoundEnd { game_id: 0 }
RoundStart { game_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 4 }, Card { suit: 3, value: 11 }], total_value: HandValue { soft: 15, hard: 15 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 12 }], total_value: HandValue { soft: 12, hard: 12 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "****", bust: "******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 2, value: 4 }, Card { suit: 3, value: 11 }], total_value: HandValue { soft: 15, hard: 15 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 1, value: 6 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 2, value: 12 }, Card { suit: 1, value: 1 }, Card { suit: 1, value: 6 }], total_value: HandValue { soft: 19, hard: 29 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 15, dealer_score: 19 } }
RoundEnd { game_id: 0 }
RoundStart { game_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 8 }, Card { suit: 1, value: 8 }], total_value: HandValue { soft: 16, hard: 16 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 11 }], total_value: HandValue { soft: 11, hard: 11 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "*****", bust: "*****" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 0, value: 8 }, Card { suit: 1, value: 8 }], total_value: HandValue { soft: 16, hard: 16 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 1, value: 10 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 11 }, Card { suit: 3, value: 3 }, Card { suit: 1, value: 10 }], total_value: HandValue { soft: 24, hard: 24 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 24 } }
Info { game_id: 0, event: Wins { player: User, user_score: 16, dealer_score: 24 } }
RoundEnd { game_id: 0 }
RoundStart { game_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 5 }, Card { suit: 1, value: 2 }], total_value: HandValue { soft: 7, hard: 7 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 2 }], total_value: HandValue { soft: 2, hard: 2 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 0, value: 1 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "**********", bust: "" } }
Action { game_id: 0, event: Hits { player: User, card: Card { suit: 3, value: 6 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "****", bust: "******" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 5 }, Card { suit: 1, value: 2 }, Card { suit: 0, value: 1 }, Card { suit: 3, value: 6 }], total_value: HandValue { soft: 14, hard: 24 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 9 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 8 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 0, value: 2 }, Card { suit: 1, value: 3 }, Card { suit: 3, value: 9 }, Card { suit: 3, value: 8 }], total_value: HandValue { soft: 22, hard: 22 } } }
Info { game_id: 0, event: Wins { player: Dealer, user_score: 14, dealer_score: 22 } }
RoundEnd { game_id: 0 }
RoundStart { game_id: 0 }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 12 }, Card { suit: 2, value: 13 }], total_value: HandValue { soft: 25, hard: 25 } } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 9 }], total_value: HandValue { soft: 9, hard: 9 } } }
Info { game_id: 0, event: Probabilities { player: User, improvement: "", bust: "**********" } }
Action { game_id: 0, event: Stands { player: User } }
Info { game_id: 0, event: Hand { player: User, cards: [Card { suit: 1, value: 12 }, Card { suit: 2, value: 13 }], total_value: HandValue { soft: 25, hard: 25 } } }
Action { game_id: 0, event: Hits { player: Dealer, card: Card { suit: 3, value: 13 } } }
Action { game_id: 0, event: Stands { player: Dealer } }
Info { game_id: 0, event: Hand { player: Dealer, cards: [Card { suit: 1, value: 9 }, Card { suit: 3, value: 1 }, Card { suit: 3, value: 13 }], total_value: HandValue { soft: 23, hard: 33 } } }
Info { game_id: 0, event: Busts { player: Dealer, score: 23 } }
Info { game_id: 0, event: Busts { player: User, score: 25 } }
RoundTied { game_id: 0, user_score: 25, dealer_score: 23 }
RoundEnd { game_id: 0 }
GameEnd { game_id: 0 }
EndSession
```

</details>

## Decoded Data (after tokenization)

We get rid of all unnecessary tokens and only keep the relevant ones reduce time to train the model. For details see `src/tokenizer.rs`.

<details>
<summary>Click to expand</summary>

```
BeginSession
GameStart game_id: 0 num_suits: 4 num_cards_per_suit: 1 3 num_decks: 1 target_score: 2 2 dealer_threshold: 1 8 
RoundStart game_id: 0 
game_id: 0 Hand player:User card: suit: 0 value: 7 suit: 3 value: 4 total_value: soft: 1 1 hard: 1 1 
game_id: 0 Hand player:Dealer card: suit: 0 value: 3 total_value: soft: 3 hard: 3 
game_id: 0 Probabilities improvement: * * * * * * * * bust: * * 
game_id: 0 Hits player:User card: suit: 1 value: 7 
game_id: 0 Probabilities improvement: * * * bust: * * * * * * * 
game_id: 0 Stands player:User 
game_id: 0 Hand player:User card: suit: 0 value: 7 suit: 3 value: 4 suit: 1 value: 7 total_value: soft: 1 8 hard: 1 8 
game_id: 0 Hits player:Dealer card: suit: 2 value: 7 
game_id: 0 Stands player:Dealer 
game_id: 0 Hand player:Dealer card: suit: 0 value: 3 suit: 3 value: 1 2 suit: 2 value: 7 total_value: soft: 2 2 hard: 2 2 
game_id: 0 Wins player:Dealer user_score: 1 8 dealer_score: 2 2 
RoundEnd game_id: 0 
RoundStart game_id: 0 
game_id: 0 Hand player:User card: suit: 0 value: 1 1 suit: 2 value: 2 total_value: soft: 1 3 hard: 1 3 
game_id: 0 Hand player:Dealer card: suit: 3 value: 2 total_value: soft: 2 hard: 2 
game_id: 0 Probabilities improvement: * * * * * * * bust: * * * 
game_id: 0 Hits player:User card: suit: 1 value: 4 
game_id: 0 Probabilities improvement: * * * bust: * * * * * * * 
game_id: 0 Stands player:User 
game_id: 0 Hand player:User card: suit: 0 value: 1 1 suit: 2 value: 2 suit: 1 value: 4 total_value: soft: 1 7 hard: 1 7 
game_id: 0 Hits player:Dealer card: suit: 3 value: 1 0 
game_id: 0 Stands player:Dealer 
game_id: 0 Hand player:Dealer card: suit: 3 value: 2 suit: 2 value: 1 1 suit: 3 value: 1 0 total_value: soft: 2 3 hard: 2 3 
game_id: 0 Busts player:Dealer score: 2 3 
game_id: 0 Wins player:User user_score: 1 7 dealer_score: 2 3 
RoundEnd game_id: 0 
RoundStart game_id: 0 
game_id: 0 Hand player:User card: suit: 0 value: 1 0 suit: 0 value: 4 total_value: soft: 1 4 hard: 1 4 
game_id: 0 Hand player:Dealer card: suit: 2 value: 5 total_value: soft: 5 hard: 5 
game_id: 0 Probabilities improvement: * * * * * * bust: * * * * 
game_id: 0 Hits player:User card: suit: 0 value: 1 2 
game_id: 0 Probabilities improvement: bust: * * * * * * * * * * 
game_id: 0 Stands player:User 
game_id: 0 Hand player:User card: suit: 0 value: 1 0 suit: 0 value: 4 suit: 0 value: 1 2 total_value: soft: 2 6 hard: 2 6 
game_id: 0 Stands player:Dealer 
game_id: 0 Hand player:Dealer card: suit: 2 value: 5 suit: 1 value: 1 3 total_value: soft: 1 8 hard: 1 8 
game_id: 0 Busts player:User score: 2 6 
game_id: 0 Wins player:Dealer user_score: 2 6 dealer_score: 1 8 
RoundEnd game_id: 0 
RoundStart game_id: 0 
game_id: 0 Hand player:User card: suit: 0 value: 9 suit: 3 value: 5 total_value: soft: 1 4 hard: 1 4 
game_id: 0 Hand player:Dealer card: suit: 2 value: 8 total_value: soft: 8 hard: 8 
game_id: 0 Probabilities improvement: * * * * * * bust: * * * * 
game_id: 0 Hits player:User card: suit: 0 value: 6 
game_id: 0 Probabilities improvement: * * bust: * * * * * * * * 
game_id: 0 Stands player:User 
game_id: 0 Hand player:User card: suit: 0 value: 9 suit: 3 value: 5 suit: 0 value: 6 total_value: soft: 2 0 hard: 2 0 
game_id: 0 Stands player:Dealer 
game_id: 0 Hand player:Dealer card: suit: 2 value: 8 suit: 0 value: 1 3 total_value: soft: 2 1 hard: 2 1 
game_id: 0 Wins player:Dealer user_score: 2 0 dealer_score: 2 1 
RoundEnd game_id: 0 
RoundStart game_id: 0 
game_id: 0 Hand player:User card: suit: 0 value: 5 suit: 3 value: 7 total_value: soft: 1 2 hard: 1 2 
game_id: 0 Hand player:Dealer card: suit: 2 value: 9 total_value: soft: 9 hard: 9 
game_id: 0 Probabilities improvement: * * * * * * * bust: * * * 
game_id: 0 Hits player:User card: suit: 2 value: 3 
game_id: 0 Probabilities improvement: * * * * bust: * * * * * * 
game_id: 0 Stands player:User 
game_id: 0 Hand player:User card: suit: 0 value: 5 suit: 3 value: 7 suit: 2 value: 3 total_value: soft: 1 5 hard: 1 5 
game_id: 0 Stands player:Dealer 
game_id: 0 Hand player:Dealer card: suit: 2 value: 9 suit: 2 value: 1 0 total_value: soft: 1 9 hard: 1 9 
game_id: 0 Wins player:Dealer user_score: 1 5 dealer_score: 1 9 
RoundEnd game_id: 0 
RoundStart game_id: 0 
game_id: 0 Hand player:User card: suit: 2 value: 4 suit: 3 value: 1 1 total_value: soft: 1 5 hard: 1 5 
game_id: 0 Hand player:Dealer card: suit: 2 value: 1 2 total_value: soft: 1 2 hard: 1 2 
game_id: 0 Probabilities improvement: * * * * bust: * * * * * * 
game_id: 0 Stands player:User 
game_id: 0 Hand player:User card: suit: 2 value: 4 suit: 3 value: 1 1 total_value: soft: 1 5 hard: 1 5 
game_id: 0 Hits player:Dealer card: suit: 1 value: 6 
game_id: 0 Stands player:Dealer 
game_id: 0 Hand player:Dealer card: suit: 2 value: 1 2 suit: 1 value: 1 suit: 1 value: 6 total_value: soft: 1 9 hard: 2 9 
game_id: 0 Wins player:Dealer user_score: 1 5 dealer_score: 1 9 
RoundEnd game_id: 0 
RoundStart game_id: 0 
game_id: 0 Hand player:User card: suit: 0 value: 8 suit: 1 value: 8 total_value: soft: 1 6 hard: 1 6 
game_id: 0 Hand player:Dealer card: suit: 1 value: 1 1 total_value: soft: 1 1 hard: 1 1 
game_id: 0 Probabilities improvement: * * * * * bust: * * * * * 
game_id: 0 Stands player:User 
game_id: 0 Hand player:User card: suit: 0 value: 8 suit: 1 value: 8 total_value: soft: 1 6 hard: 1 6 
game_id: 0 Hits player:Dealer card: suit: 1 value: 1 0 
game_id: 0 Stands player:Dealer 
game_id: 0 Hand player:Dealer card: suit: 1 value: 1 1 suit: 3 value: 3 suit: 1 value: 1 0 total_value: soft: 2 4 hard: 2 4 
game_id: 0 Busts player:Dealer score: 2 4 
game_id: 0 Wins player:User user_score: 1 6 dealer_score: 2 4 
RoundEnd game_id: 0 
RoundStart game_id: 0 
game_id: 0 Hand player:User card: suit: 1 value: 5 suit: 1 value: 2 total_value: soft: 7 hard: 7 
game_id: 0 Hand player:Dealer card: suit: 0 value: 2 total_value: soft: 2 hard: 2 
game_id: 0 Probabilities improvement: * * * * * * * * * * bust: 
game_id: 0 Hits player:User card: suit: 0 value: 1 
game_id: 0 Probabilities improvement: * * * * * * * * * * bust: 
game_id: 0 Hits player:User card: suit: 3 value: 6 
game_id: 0 Probabilities improvement: * * * * bust: * * * * * * 
game_id: 0 Stands player:User 
game_id: 0 Hand player:User card: suit: 1 value: 5 suit: 1 value: 2 suit: 0 value: 1 suit: 3 value: 6 total_value: soft: 1 4 hard: 2 4 
game_id: 0 Hits player:Dealer card: suit: 3 value: 9 
game_id: 0 Hits player:Dealer card: suit: 3 value: 8 
game_id: 0 Stands player:Dealer 
game_id: 0 Hand player:Dealer card: suit: 0 value: 2 suit: 1 value: 3 suit: 3 value: 9 suit: 3 value: 8 total_value: soft: 2 2 hard: 2 2 
game_id: 0 Wins player:Dealer user_score: 1 4 dealer_score: 2 2 
RoundEnd game_id: 0 
RoundStart game_id: 0 
game_id: 0 Hand player:User card: suit: 1 value: 1 2 suit: 2 value: 1 3 total_value: soft: 2 5 hard: 2 5 
game_id: 0 Hand player:Dealer card: suit: 1 value: 9 total_value: soft: 9 hard: 9 
game_id: 0 Probabilities improvement: bust: * * * * * * * * * * 
game_id: 0 Stands player:User 
game_id: 0 Hand player:User card: suit: 1 value: 1 2 suit: 2 value: 1 3 total_value: soft: 2 5 hard: 2 5 
game_id: 0 Hits player:Dealer card: suit: 3 value: 1 3 
game_id: 0 Stands player:Dealer 
game_id: 0 Hand player:Dealer card: suit: 1 value: 9 suit: 3 value: 1 suit: 3 value: 1 3 total_value: soft: 2 3 hard: 3 3 
game_id: 0 Busts player:Dealer score: 2 3 
game_id: 0 Busts player:User score: 2 5 
RoundTied game_id: 0 user_score: 2 5 dealer_score: 2 3 
RoundEnd game_id: 0 
GameEnd game_id: 0 
EndSession
```

</details>

## Example vocabulary

<details>
<summary>Click to expand</summary>

```json
{
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[PAD]": 0,
      " ": 1,
      "*": 2,
      "0": 3,
      "\n": 4,
      "game_id:": 5,
      "1": 6,
      "suit:": 7,
      "value:": 8,
      "2": 9,
      "card:": 10,
      "3": 11,
      "player:Dealer": 12,
      "player:User": 13,
      "Hand": 14,
      "hard:": 15,
      "soft:": 16,
      "total_value:": 17,
      "5": 18,
      "4": 19,
      "9": 20,
      "Stands": 21,
      "8": 22,
      "Probabilities": 23,
      "bust:": 24,
      "improvement:": 25,
      "6": 26,
      "Hits": 27,
      "7": 28,
      "RoundEnd": 29,
      "RoundStart": 30,
      "dealer_score:": 31,
      "user_score:": 32,
      "Wins": 33,
      "Busts": 34,
      "score:": 35,
      "BeginSession": 36,
      "EndSession": 37,
      "GameEnd": 38,
      "GameStart": 39,
      "RoundTied": 40,
      "dealer_threshold:": 41,
      "num_cards_per_suit:": 42,
      "num_decks:": 43,
      "num_suits:": 44,
      "target_score:": 45
    },
    "unk_token": "[UNK]"
  }
}
```

</details>
