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
```

The winrates are:

```
User wins: 1339 (43.99%)
Dealer wins: 1147 (37.68%)
Ties: 558 (18.33%)
```
