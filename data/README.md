# Data Directory

## Structure

```
data/
├── tiny/           <- small CSV files included in the repository
│   ├── tiny_sentiment.csv          <- 90 labeled sentences (positive/negative/neutral)
│   ├── tiny_translation_en_fr.csv  <- 100 English-French sentence pairs
│   └── README.md
├── raw/            <- downloaded datasets (gitignored, auto-downloaded by torchvision)
└── processed/      <- preprocessed data (gitignored)
```

## tiny/

These files are small, hand-crafted datasets included directly in the repository. No download required.

## raw/

This folder is gitignored. Torchvision datasets (Fashion-MNIST, CIFAR-10, Oxford-IIIT Pet) are downloaded here automatically when first used.

## processed/

This folder is gitignored. Preprocessed or cached data may be stored here.
