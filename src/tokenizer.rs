use tokenizers::{
    models::wordlevel::{
        WordLevel,
        WordLevelTrainerBuilder,
    },
    normalizers::{
        strip::Strip,
        utils::Sequence,
        Replace,
    },
    pre_tokenizers::{
        byte_level::ByteLevel,
        split::{
            Split,
            SplitPattern,
        },
    },
    utils::padding::PaddingParams,
    AddedToken,
    NormalizerWrapper,
    PaddingDirection,
    PaddingStrategy,
    Result,
    TokenizerBuilder,
    TokenizerImpl,
};

pub fn create_tokenizer(
    transcript_file: &str,
    output_prefix: String,
) -> Result<TokenizerImpl<WordLevel, Sequence, Split, ByteLevel, ByteLevel>> {
    let mut trainer = WordLevelTrainerBuilder::default()
        .show_progress(true)
        .vocab_size(9999)
        .special_tokens(vec![AddedToken::from("[PAD]", true).single_word(true)])
        .min_frequency(0)
        .build()?;

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(WordLevel::builder().unk_token("[UNK]".into()).build()?)
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(false, false).into(),
            NormalizerWrapper::Replace(Replace::new("*", " * ")?),
            NormalizerWrapper::Replace(Replace::new("0", " 0 ")?),
            NormalizerWrapper::Replace(Replace::new("1", " 1 ")?),
            NormalizerWrapper::Replace(Replace::new("2", " 2 ")?),
            NormalizerWrapper::Replace(Replace::new("3", " 3 ")?),
            NormalizerWrapper::Replace(Replace::new("4", " 4 ")?),
            NormalizerWrapper::Replace(Replace::new("5", " 5 ")?),
            NormalizerWrapper::Replace(Replace::new("6", " 6 ")?),
            NormalizerWrapper::Replace(Replace::new("7", " 7 ")?),
            NormalizerWrapper::Replace(Replace::new("8", " 8 ")?),
            NormalizerWrapper::Replace(Replace::new("9", " 9 ")?),
            NormalizerWrapper::Replace(Replace::new(",", "")?),
            NormalizerWrapper::Replace(Replace::new("event:", "")?),
            NormalizerWrapper::Replace(Replace::new("player: User", "player:User")?),
            NormalizerWrapper::Replace(Replace::new("player: Dealer", "player:Dealer")?),
            NormalizerWrapper::Replace(Replace::new("Action { ", "")?),
            NormalizerWrapper::Replace(Replace::new("Info { ", "")?),
            NormalizerWrapper::Replace(Replace::new("Card ", "")?),
            NormalizerWrapper::Replace(Replace::new("HandValue ", "")?),
            NormalizerWrapper::Replace(Replace::new("cards: ", "card: ")?),
            NormalizerWrapper::Replace(Replace::new("{", "")?),
            NormalizerWrapper::Replace(Replace::new("}", "")?),
            NormalizerWrapper::Replace(Replace::new("[", "")?),
            NormalizerWrapper::Replace(Replace::new("]", "")?),
            NormalizerWrapper::Replace(Replace::new("\"", "")?),
            NormalizerWrapper::Replace(Replace::new("     ", " ")?),
            NormalizerWrapper::Replace(Replace::new("    ", " ")?),
            NormalizerWrapper::Replace(Replace::new("   ", " ")?),
            NormalizerWrapper::Replace(Replace::new("  ", " ")?),
            NormalizerWrapper::Replace(Replace::new("\n ", "\n")?),
            NormalizerWrapper::Replace(Replace::new(" \n", "\n")?),
            NormalizerWrapper::Replace(Replace::new(
                "Probabilities player:User ",
                "Probabilities ",
            )?),
        ])))
        .with_pre_tokenizer(Some(Split::new(
            SplitPattern::Regex(r"\s".to_string()),
            tokenizers::SplitDelimiterBehavior::Isolated,
            false,
        )?))
        .with_post_processor(Some(ByteLevel::default().add_prefix_space(false)))
        .with_decoder(Some(ByteLevel::default().add_prefix_space(false)))
        .with_padding(Some(PaddingParams {
            strategy:           PaddingStrategy::BatchLongest,
            direction:          PaddingDirection::Left,
            pad_to_multiple_of: Some(8),
            pad_id:             0,
            pad_type_id:        0,
            pad_token:          String::from("[PAD]"),
        }))
        .build()?;

    let pretty = true;
    tokenizer
        .train_from_files(&mut trainer, vec![transcript_file.to_string()])?
        .save(output_prefix + "-tokenizer.json", pretty)?;

    Ok(tokenizer)
}
