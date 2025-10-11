CREATE TABLE samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT
);

CREATE TABLE tokens (
    id INTEGER NOT NULL,
    sample_id INTEGER NOT NULL,
    token_text TEXT NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES samples(id),
    PRIMARY KEY (id, sample_id)
);

CREATE TABLE activations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    sample_id INTEGER NOT NULL,
    value REAL NOT NULL,
    FOREIGN KEY (token_id) REFERENCES tokens(id)
    FOREIGN KEY (sample_id) REFERENCES samples(id)
);