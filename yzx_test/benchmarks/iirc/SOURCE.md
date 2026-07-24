# IIRC Dataset

Downloaded from the IIRC dataset storage referenced by the IRCoT raw-data
download script:

- https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
- https://iirc-dataset.s3.us-west-2.amazonaws.com/context_articles.tar.gz

Source script:

- https://github.com/StonyBrookNLP/ircot/blob/main/download/raw_data.sh

## Archive Checksums

- `iirc_train_dev.tgz`: `adfc34c8180337467105b2f534410c34e4fe43a82e6da03922440387802ca441`
- `context_articles.tar.gz`: `d239390a27ed3f4aa868afe126187e6d108990a7bcc2a3efb6cae5de917264c7`

## Extracted Data

- `train.json`: 4,754 passage records and 10,839 questions
- `dev.json`: 430 passage records and 1,301 questions
- `context_articles.json`: 56,550 linked Wikipedia articles

The public train and development splits contain 12,140 questions in total.
