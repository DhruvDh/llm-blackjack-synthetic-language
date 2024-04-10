use bpaf::Bpaf;
use futures::{
    stream::FuturesUnordered,
    StreamExt,
};
use num_format::ToFormattedString;
use tokio::{
    fs::{
        File,
        OpenOptions,
    },
    io::{
        AsyncBufReadExt,
        AsyncWriteExt,
        BufReader,
    },
};

mod blackjack;
mod tokenizer;

use crate::{
    blackjack::generate_transcripts,
    tokenizer::create_tokenizer,
};

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
    /// Prefix for output files (default: blackjack_transcripts)
    #[bpaf(long, fallback("blackjack_transcripts".to_string()))]
    output:                     String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = config().run();

    let file_path = config.output.clone() + ".txt";

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
        .open(file_path.clone())
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

    let contents = tokio::fs::read_to_string(file_path.clone())
        .await
        .expect("Unable to read file");

    let user_wins = contents.matches("Wins { player: User").count();
    let dealer_wins = contents.matches("Wins { player: Dealer").count();
    let ties = contents.matches("GameTied {").count();
    let total_games = user_wins + dealer_wins + ties;

    println!(
        "User wins:\t{} ({:.2}%)",
        user_wins.to_formatted_string(&num_format::Locale::en),
        user_wins as f64 / total_games as f64 * 100.0
    );
    println!(
        "Dealer wins:\t{} ({:.2}%)",
        dealer_wins.to_formatted_string(&num_format::Locale::en),
        dealer_wins as f64 / total_games as f64 * 100.0
    );
    println!(
        "Ties:\t\t{} ({:.2}%)",
        ties.to_formatted_string(&num_format::Locale::en),
        ties as f64 / total_games as f64 * 100.0
    );
    println!(
        "Total games:\t{}",
        total_games.to_formatted_string(&num_format::Locale::en)
    );

    println!("Creating tokenizer...");
    let tokenizer = create_tokenizer(&file_path, config.output.clone()).unwrap();

    println!("Counting tokens... (feel free to interrupt)");
    let num_threads = std::thread::available_parallelism()?.get();

    let file = File::open(file_path).await?;
    let mut num_tokens = 0;
    let mut reader = BufReader::new(file);

    'outer: loop {
        for _ in 0..(num_threads * 1024) {
            let mut batch = Vec::with_capacity(num_threads);
            let mut buffer = String::new();
            let line = reader.read_line(&mut buffer).await?;
            if line == 0 {
                break 'outer;
            }
            batch.push(buffer.clone());

            let encoding = tokenizer.encode_batch(batch, false).unwrap_or_default();

            for enc in encoding {
                num_tokens += enc.get_ids().len();
            }
        }
    }

    println!(
        "Number of tokens: {}",
        num_tokens.to_formatted_string(&num_format::Locale::en)
    );
    Ok(())
}
