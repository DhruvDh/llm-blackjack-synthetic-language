use tokenizers::{
    models::bpe::{
        BpeTrainerBuilder,
        BPE,
    },
    normalizers::{
        strip::Strip,
        utils::Sequence,
        Lowercase,
    },
    pre_tokenizers::{
        byte_level::ByteLevel,
        split::{
            Split,
            SplitPattern,
        },
    },
    NormalizerWrapper,
    Result,
    TokenizerBuilder,
};

pub fn create_tokenizer(transcript_file: &str, output_prefix: String) -> Result<()> {
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(9999)
        .min_frequency(0)
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(
            BPE::builder()
                .unk_token("<UNK>".into())
                .continuing_subword_prefix(String::new())
                .end_of_word_suffix(String::new())
                .build()?,
        )
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(false, false).into(),
            NormalizerWrapper::Lowercase(Lowercase),
        ])))
        .with_pre_tokenizer(Some(Split::new(
            SplitPattern::Regex(r"\s".to_string()),
            tokenizers::SplitDelimiterBehavior::Isolated,
            false,
        )?))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    let pretty = true;
    tokenizer
        .train_from_files(&mut trainer, vec![transcript_file.to_string()])?
        .save(output_prefix + "-tokenizer.json", pretty)?;

    Ok(())
}
