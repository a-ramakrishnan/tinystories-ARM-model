# Public text datasets that work with this project out of the box.
# Each was checked with data.inspect_source(): it loads without a script, isn't gated, and its text column holds strings.
# Sizes are download sizes reported by Hugging Face (rounded).

DATASET_CATALOG = [
    {
        "title": "Mini FineWeb-Edu",
        "dataset": "BaoMinhTran/mini-fineweb-edu", "config": "small", "text_column": "text",
        "description": "Educational web pages from FineWeb-Edu (Common Crawl CC-MAIN-2025-26), filtered for "
                       "teaching value. The 'small' subset holds ~500M tokens; 'debug', 'tiny', 'base' and 'full' "
                       "range from 10M to 2.9B. Harder than stories, so expect a higher loss.",
        "tags": ["Web", "Education", "Default"], "rows": "595K pages", "size": "1.3 GB",
        "url": "https://huggingface.co/datasets/BaoMinhTran/mini-fineweb-edu",
    },
    {
        "title": "TinyStories",
        "dataset": "roneneldan/TinyStories", "config": None, "text_column": "text",
        "description": "Short, simple stories written by GPT-3.5 and GPT-4 using words a young child knows. "
                       "The dataset this project started with; small models learn it quickly.",
        "tags": ["Stories"], "rows": "2.1M stories", "size": "1.0 GB",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
    },
    {
        "title": "Tiny Shakespeare",
        "dataset": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "config": None, "text_column": "text",
        "description": "A 1 MB selection of Shakespeare's plays in one text file. Tiny, so tokenizing and a quick "
                       "training run take minutes. Loaded from a link.",
        "tags": ["Plays", "Tiny", "Link"], "rows": "1 text file", "size": "1.1 MB",
        "url": "https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare",
    },
    {
        "title": "WikiText-2",
        "dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "text_column": "text",
        "description": "Good and featured Wikipedia articles. A classic small language modelling benchmark.",
        "tags": ["Wikipedia", "Small"], "rows": "37K lines", "size": "7.7 MB",
        "url": "https://huggingface.co/datasets/Salesforce/wikitext",
    },
    {
        "title": "WikiText-103",
        "dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "text_column": "text",
        "description": "The larger version of WikiText: over 100 million tokens of Wikipedia articles.",
        "tags": ["Wikipedia"], "rows": "1.8M lines", "size": "315 MB",
        "url": "https://huggingface.co/datasets/Salesforce/wikitext",
    },
    {
        "title": "Simple English Wikipedia",
        "dataset": "wikimedia/wikipedia", "config": "20231101.simple", "text_column": "text",
        "description": "Every article from Simple English Wikipedia, written with plain words and short sentences.",
        "tags": ["Wikipedia", "Simple English"], "rows": "242K articles", "size": "157 MB",
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
    },
    {
        "title": "WritingPrompts",
        "dataset": "euclaise/writingprompts", "config": None, "text_column": "story",
        "description": "Stories written by Reddit users in reply to creative writing prompts. Longer and more "
                       "grown-up than TinyStories.",
        "tags": ["Stories"], "rows": "273K stories", "size": "605 MB",
        "url": "https://huggingface.co/datasets/euclaise/writingprompts",
    },
    {
        "title": "TinyStories Instruct",
        "dataset": "roneneldan/TinyStoriesInstruct", "config": None, "text_column": "text",
        "description": "TinyStories with instructions: each story comes with the words, features or summary it "
                       "was written to include.",
        "tags": ["Stories", "Large"], "rows": "21.8M lines", "size": "2.7 GB",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct",
    },
    {
        "title": "CNN / DailyMail",
        "dataset": "abisee/cnn_dailymail", "config": "3.0.0", "text_column": "article",
        "description": "News articles from CNN and the Daily Mail. Trains on the article text.",
        "tags": ["News"], "rows": "287K articles", "size": "837 MB",
        "url": "https://huggingface.co/datasets/abisee/cnn_dailymail",
    },
    {
        "title": "AG News",
        "dataset": "fancyzhx/ag_news", "config": None, "text_column": "text",
        "description": "Short news headlines and descriptions in four topics: world, sports, business, science.",
        "tags": ["News", "Small"], "rows": "120K articles", "size": "20 MB",
        "url": "https://huggingface.co/datasets/fancyzhx/ag_news",
    },
    {
        "title": "IMDB Reviews",
        "dataset": "stanfordnlp/imdb", "config": None, "text_column": "text",
        "description": "Movie reviews from IMDB. Informal, opinionated writing.",
        "tags": ["Reviews"], "rows": "25K reviews", "size": "83 MB",
        "url": "https://huggingface.co/datasets/stanfordnlp/imdb",
    },
    {
        "title": "Rotten Tomatoes",
        "dataset": "cornell-movie-review-data/rotten_tomatoes", "config": None, "text_column": "text",
        "description": "One-sentence movie review snippets. Very small: good for checking the pipeline end to end.",
        "tags": ["Reviews", "Tiny"], "rows": "8.5K sentences", "size": "0.5 MB",
        "url": "https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes",
    },
]
