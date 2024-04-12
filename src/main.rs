use bpaf::Bpaf;
use futures::{
    stream::FuturesUnordered,
    StreamExt,
};
use num_format::ToFormattedString;
use tokio::{
    fs::OpenOptions,
    io::AsyncWriteExt,
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
async fn process_transcripts(config: &Config) -> anyhow::Result<String> {
    let file_path = config.output.clone() + ".txt";
    let mut output = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file_path.clone())
        .await?;

    let mut tasks = FuturesUnordered::new();
    for num_decks in config.min_decks..=config.max_decks {
        for num_suits in config.min_suits..=config.max_suits {
            for num_cards_per_suit in config.min_cards_per_suit..=config.max_cards_per_suit {
                for num_simultaneous_games in 1..=config.max_num_simultaneous_games {
                    for _ in 0..config.num_games {
                        let task = tokio::spawn(async move {
                            generate_transcripts(
                                num_suits,
                                num_cards_per_suit,
                                num_decks,
                                num_simultaneous_games,
                            )
                        });
                        tasks.push(task);
                    }
                }
            }
        }
    }

    let progress = indicatif::ProgressBar::new(tasks.len() as u64);
    while let Some(result) = tasks.next().await {
        let transcript = result?;
        output.write_all(transcript.as_bytes()).await?;
        output.write_all(b"\n").await?;
        progress.inc(1);
    }

    Ok(file_path)
}

async fn analyze_transcripts(file_path: &str) -> anyhow::Result<()> {
    let contents = tokio::fs::read_to_string(file_path).await?;

    let user_wins = contents.matches("Wins { player: User").count();
    let dealer_wins = contents.matches("Wins { player: Dealer").count();
    let ties = contents.matches("RoundTied {").count();
    let total_rounds = user_wins + dealer_wins + ties;
    let total_games = contents.matches("GameEnd {").count();
    let total_sessions = contents.matches("EndSession").count();

    println!(
        "User wins:\t{} ({:.2}%)",
        user_wins.to_formatted_string(&num_format::Locale::en),
        user_wins as f64 / total_rounds as f64 * 100.0
    );
    println!(
        "Dealer wins:\t{} ({:.2}%)",
        dealer_wins.to_formatted_string(&num_format::Locale::en),
        dealer_wins as f64 / total_rounds as f64 * 100.0
    );
    println!(
        "Ties:\t\t{} ({:.2}%)",
        ties.to_formatted_string(&num_format::Locale::en),
        ties as f64 / total_rounds as f64 * 100.0
    );
    println!(
        "Total rounds:\t{}",
        total_rounds.to_formatted_string(&num_format::Locale::en)
    );
    println!(
        "Total games:\t{}",
        total_games.to_formatted_string(&num_format::Locale::en)
    );
    println!(
        "Total sessions:\t{}",
        total_sessions.to_formatted_string(&num_format::Locale::en)
    );

    Ok(())
}

async fn tokenize_and_decode(file_path: &str, output_prefix: &str) -> anyhow::Result<()> {
    let contents = tokio::fs::read_to_string(file_path).await?;
    let lines = contents
        .split('\n')
        .map(|l| l.to_string())
        .collect::<Vec<String>>();

    println!("Creating tokenizer...");
    let tokenizer = create_tokenizer(file_path, output_prefix.to_string()).unwrap();

    println!("Tokenizing...");
    let instant = std::time::Instant::now();
    let encoding = tokenizer.encode_batch(lines, true).unwrap();
    let pad_id = tokenizer.get_padding().unwrap().pad_id;
    let num_tokens = encoding
        .iter()
        .map(|e| e.get_ids().iter().filter(|&&id| id != pad_id).count())
        .sum::<usize>();

    println!(
        "Number of tokens:\t\t{}",
        num_tokens.to_formatted_string(&num_format::Locale::en)
    );
    let token_per_second = num_tokens as f64 / instant.elapsed().as_secs_f64();
    println!("Tokens encoded per second:\t{:.2}", token_per_second);

    let total_sessions = contents.matches("EndSession").count();
    println!(
        "Average tokens per session:\t{:.2}",
        num_tokens as f64 / total_sessions as f64
    );

    let sessions = contents.split("EndSession");
    let mut longest_session = 0;
    for session in sessions {
        let session_lines = session
            .trim()
            .split('\n')
            .map(|l| l.to_string())
            .collect::<Vec<String>>();
        let session_encoding = tokenizer.encode_batch(session_lines, true).unwrap();
        let session_tokens = session_encoding
            .iter()
            .map(|e| e.get_ids().iter().filter(|&&id| id != pad_id).count())
            .sum::<usize>();
        longest_session = longest_session.max(session_tokens);
    }
    longest_session += 1; // Add 1 for the EndSession token

    println!(
        "Longest session (tokens):\t{}",
        longest_session.to_formatted_string(&num_format::Locale::en)
    );

    println!("Decoding...");
    let decoded_lines = encoding
        .into_iter()
        .map(|e| tokenizer.decode(e.get_ids(), true).unwrap())
        .collect::<Vec<String>>();

    let decoded_output_path = format!("{}_decoded.txt", output_prefix);
    let mut decoded_output = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(decoded_output_path)
        .await?;
    for line in decoded_lines {
        decoded_output.write_all(line.as_bytes()).await?;
        decoded_output.write_all(b"\n").await?;
    }

    Ok(())
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = config().run();
    println!(
        "{}",
        blackjack::generate_target_score_table(
            config.min_suits,
            config.max_suits,
            config.min_cards_per_suit,
            config.max_cards_per_suit,
        )
    );

    let file_path = process_transcripts(&config).await?;
    analyze_transcripts(&file_path).await?;
    tokenize_and_decode(&file_path, &config.output).await?;

    Ok(())
}
