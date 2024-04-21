use std::collections::HashMap;

use anyhow::Context;
use blackjack::{
    Event,
    GameID,
    InfoEvent,
};
use bpaf::Bpaf;
use futures::{
    stream::FuturesUnordered,
    StreamExt,
};
use num_format::ToFormattedString;
use rand::seq::SliceRandom;
use serde::{
    Deserialize,
    Serialize,
};
use tokio::{
    fs::OpenOptions,
    io::AsyncWriteExt,
};
use tokio_stream::wrappers::ReadDirStream;

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
    /// Maximum number of decks (default: 1)
    #[bpaf(long, fallback(1))]
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
    /// Maximum number of cards per suit (default: 15)
    #[bpaf(long, fallback(15))]
    max_cards_per_suit:         usize,
    /// Number of games to generate for each configuration (default: 33)
    #[bpaf(short, long, fallback(33))]
    num_games:                  usize,
    /// Maximum number of simultaneous games to generate (default: 2)
    #[bpaf(short, long, fallback(2))]
    max_num_simultaneous_games: usize,
    /// Number of games to generate for the eval set. (default: 6)
    #[bpaf(long, fallback(6))]
    num_eval_games:             usize,
    /// Output directory for the generated data. (default: data)
    #[bpaf(long, fallback("data-generated-icl".to_string()))]
    output_dir:                 String,
}

async fn process_transcripts(
    config: &Config,
    output_dir: &str,
    num_games: usize,
) -> anyhow::Result<Vec<String>> {
    tokio::fs::create_dir_all(output_dir)
        .await
        .context("Failed to create output directory")?;

    let mut tasks = FuturesUnordered::new();
    let mut game_id = 0;

    for _ in 0..num_games {
        for num_decks in config.min_decks..=config.max_decks {
            for num_suits in config.min_suits..=config.max_suits {
                for num_cards_per_suit in config.min_cards_per_suit..=config.max_cards_per_suit {
                    for num_simultaneous_games in 1..=config.max_num_simultaneous_games {
                        let output_dir = output_dir.to_string();
                        let task = tokio::spawn(async move {
                            let transcript = generate_transcripts(
                                num_suits,
                                num_cards_per_suit,
                                num_decks,
                                num_simultaneous_games,
                            );
                            let file_path = format!("{}/session_{}.txt", output_dir, game_id);
                            let mut file = OpenOptions::new()
                                .create(true)
                                .write(true)
                                .truncate(true)
                                .open(file_path.clone())
                                .await?;
                            file.write_all(transcript.as_bytes()).await?;
                            Ok::<_, anyhow::Error>(file_path)
                        });
                        tasks.push(task);
                        game_id += 1;
                    }
                }
            }
        }
    }

    let progress = indicatif::ProgressBar::new(tasks.len() as u64);
    let mut file_paths = Vec::new();
    while let Some(result) = tasks.next().await {
        file_paths.push(result??);
        progress.inc(1);
    }

    Ok(file_paths)
}

async fn analyze_transcripts(file_paths: &[String], label: &str) -> anyhow::Result<()> {
    let mut contents = String::new();
    for file_path in file_paths {
        contents.push_str(
            &tokio::fs::read_to_string(file_path)
                .await
                .with_context(|| format!("Failed to read file: {}", file_path))?,
        );
        contents.push('\n');
    }

    let user_wins = contents.matches("event:Wins(player:User").count();
    let dealer_wins = contents.matches("event:Wins(player:Dealer").count();
    let ties = contents.matches("RoundTied(game_id:").count();
    let total_rounds = contents.matches("RoundEnd(game_id:").count();
    let total_games = contents.matches("GameEnd(game_id:").count();
    let total_sessions = contents.matches("EndSession").count();

    println!("Results for {}:", label);
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
    println!();

    Ok(())
}

async fn tokenize_and_decode(file_paths: &[String], output_prefix: &str) -> anyhow::Result<()> {
    let mut contents = String::new();
    for file_path in file_paths {
        contents.push_str(
            &tokio::fs::read_to_string(file_path)
                .await
                .with_context(|| format!("Failed to read file: {}", file_path))?,
        );
        contents.push('\n');
    }

    let lines = contents
        .split('\n')
        .map(|l| l.to_string())
        .collect::<Vec<String>>();

    println!("Creating tokenizer for {}...", output_prefix);
    let temp_file_path = format!("{}_temp.txt", output_prefix);
    let mut temp_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&temp_file_path)
        .await?;
    temp_file.write_all(contents.as_bytes()).await?;
    let tokenizer = create_tokenizer(&temp_file_path, output_prefix.to_string()).unwrap();
    tokio::fs::remove_file(&temp_file_path).await?;

    println!("Tokenizing {}...", output_prefix);
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

    println!("Decoding {}...", output_prefix);
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

async fn shuffle_files(directory: &str) -> anyhow::Result<()> {
    let temp_dir = format!("{}_temp", directory);
    tokio::fs::create_dir_all(&temp_dir).await?;

    let mut files = tokio::fs::read_dir(directory).await?;
    let mut file_names = Vec::new();

    while let Some(entry) = files.next_entry().await? {
        let file_name = entry.file_name().to_string_lossy().into_owned();
        let old_path = format!("{}/{}", directory, file_name);
        let new_path = format!("{}/{}", temp_dir, file_name);
        tokio::fs::rename(&old_path, &new_path).await?;
        file_names.push(file_name);
    }

    let mut rng = rand::thread_rng();
    file_names.shuffle(&mut rng);

    for (i, file_name) in file_names.iter().enumerate() {
        let old_path = format!("{}/{}", temp_dir, file_name);
        let new_path = format!("{}/session_{}.txt", directory, i);
        tokio::fs::rename(&old_path, &new_path).await?;
    }

    tokio::fs::remove_dir_all(&temp_dir).await?;

    Ok(())
}

fn generate_all_configs(config: &Config) -> Vec<(usize, usize, usize)> {
    let mut configs = Vec::new();

    for num_decks in config.min_decks..=config.max_decks {
        for num_suits in config.min_suits..=config.max_suits {
            for num_cards_per_suit in config.min_cards_per_suit..=config.max_cards_per_suit {
                configs.push((num_suits, num_cards_per_suit, num_decks));
            }
        }
    }

    configs
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct JsonlEntry {
    context:      String,
    continuation: String,
}

async fn generate_icl_eval_data(eval_dir: &str) -> anyhow::Result<()> {
    let mut eval_files_stream = ReadDirStream::new(tokio::fs::read_dir(eval_dir).await?);
    let mut tasks = FuturesUnordered::new();

    while let Some(entry) = eval_files_stream.next().await {
        let path = entry?.path();
        let task = tokio::spawn(async move {
            let file_contents = tokio::fs::read_to_string(path).await?;
            let events: Vec<Event> = file_contents
                .lines()
                .map(|line| ron::from_str(line).unwrap())
                .collect();

            let mut jsonl_entries: HashMap<String, Vec<JsonlEntry>> = HashMap::new();
            let mut game_id_to_config: HashMap<usize, (usize, usize, usize)> = HashMap::new();
            let mut session_context = String::new();

            for event in events {
                let event_string = ron::to_string(&event).unwrap();

                match event {
                    Event::BeginSession => {
                        session_context.push_str(&event_string);
                        session_context.push('\n');
                    }
                    Event::EndSession => {
                        session_context.clear();
                    }
                    Event::GameStart {
                        game_id,
                        num_suits,
                        num_cards_per_suit,
                        num_decks,
                        ..
                    } => {
                        let config = (num_suits, num_cards_per_suit, num_decks);
                        game_id_to_config.insert(game_id, config);
                    }
                    Event::Info {
                        game_id,
                        event:
                            InfoEvent::Probabilities {
                                ..
                            },
                    } => {
                        if let Some(config) = game_id_to_config.get(&game_id) {
                            let (num_suits, num_cards_per_suit, num_decks) = *config;
                            let config_key = format!(
                                "eval_suits_{}_cards_{}_decks_{}",
                                num_suits, num_cards_per_suit, num_decks
                            );
                            let (context_part, continuation_part) =
                                event_string.split_once("improvement:").unwrap();
                            let continuation = extract_stars(continuation_part);
                            let jsonl_entry = JsonlEntry {
                                context: format!("{}{}", session_context, context_part),
                                continuation,
                            };
                            jsonl_entries
                                .entry(config_key)
                                .or_default()
                                .push(jsonl_entry);
                        }
                    }
                    _ => {
                        session_context.push_str(&event_string);
                        session_context.push('\n');
                    }
                }
            }

            Ok::<_, anyhow::Error>(jsonl_entries)
        });
        tasks.push(task);
    }

    let mut all_jsonl_entries: HashMap<String, Vec<JsonlEntry>> = HashMap::new();

    while let Some(result) = tasks.next().await {
        let jsonl_entries = result??;
        for (config_key, entries) in jsonl_entries {
            all_jsonl_entries
                .entry(config_key)
                .or_default()
                .extend(entries);
        }
    }

    tokio::fs::create_dir_all("data-generated-icl/eval-icl").await?;

    let jsonl_tasks: Vec<_> = all_jsonl_entries
        .into_iter()
        .map(|(config_key, entries)| {
            tokio::spawn(async move {
                let jsonl_file_path = format!("data-generated-icl/eval-icl/{}.jsonl", config_key);
                let jsonl_file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(jsonl_file_path.clone())
                    .await?;
                let mut writer = tokio::io::BufWriter::new(jsonl_file);

                for entry in entries.clone() {
                    let json_string = serde_json::to_string(&entry).unwrap();
                    writer.write_all(json_string.as_bytes()).await?;
                    writer.write_all(b"\n").await?;
                }

                let parts: Vec<&str> = config_key.split('_').collect();
                let num_suits = parts[2].parse::<usize>().unwrap();
                let num_cards_per_suit = parts[4].parse::<usize>().unwrap();
                let num_decks = parts[6].parse::<usize>().unwrap();

                println!(
                    "Generated {} JSONL entries for configuration: ({}, {}, {})",
                    entries.len(),
                    num_suits,
                    num_cards_per_suit,
                    num_decks
                );

                Ok::<_, anyhow::Error>(())
            })
        })
        .collect();

    for task in jsonl_tasks {
        task.await??;
    }

    Ok(())
}

fn extract_stars(continuation_part: &str) -> String {
    let mut continuation = continuation_part.trim().to_string();
    if let Some(bust_index) = continuation.find("bust:") {
        continuation.truncate(bust_index);
    }
    continuation.trim().to_string()
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

    let train_dir = format!("{}/train", config.output_dir);
    let eval_dir = format!("{}/eval", config.output_dir);

    let train_file_paths = process_transcripts(&config, &train_dir, config.num_games).await?;
    let eval_file_paths = process_transcripts(&config, &eval_dir, config.num_eval_games).await?;

    let all_file_paths = [&train_file_paths[..], &eval_file_paths[..]].concat();

    analyze_transcripts(&all_file_paths, "Overall").await?;
    analyze_transcripts(&train_file_paths, "Train").await?;
    analyze_transcripts(&eval_file_paths, "Eval").await?;

    tokenize_and_decode(&all_file_paths, &format!("{}/overall", config.output_dir)).await?;

    println!("Generating eval ICL data...");
    generate_icl_eval_data(&eval_dir).await?;

    println!("Shuffling files...");
    shuffle_files(&train_dir).await?;
    shuffle_files(&eval_dir).await?;
    Ok(())
}
