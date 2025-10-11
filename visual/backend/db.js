'use strict';

const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const { execSync } = require('child_process');

const path = './sae_tokens.db';

if (!fs.existsSync(path)) {
    console.error('sae_tokens.db file not found. Creating DB using db_creation.js');
    try {
        const output = execSync('node ./db_creation.js');
        console.log(`db_creation.js: ${output.toString()}`);
        // Leggi i dati dal file JSON e inseriscili nel database
        const data = JSON.parse(fs.readFileSync('../assets/sae/token_activations.json', 'utf8'));
        const db = new sqlite3.Database(path);
        // console.log(data);
        db.serialize(() => {
            for (const sample of data) {
                // console.log(sample);
                db.run("INSERT INTO samples (id) VALUES (?)", [sample.sample_index], function(err) {
                    if (err) throw err;
                });

                const sampleId = sample.sample_index;
                console.log(`Inserted sample with ID: ${sampleId}`);

                // 2. Per ogni token
                for (const token of sample.tokens) {
                    // console.log(token);
                    db.run("INSERT INTO tokens (sample_id, token_text, id) VALUES (?, ?, ?)", [sampleId, token.token_str, token.token_idx], function(err) {
                        if (err) throw err;});
                    const tokenId = token.token_idx;
                    console.log(`  Inserted token with ID: ${tokenId} for sample ID: ${sampleId}`);

                    const stmt = db.prepare("INSERT INTO activations (token_id, feature_id, sample_id, value) VALUES (?, ?, ?, ?)", function(err) {
                        if (err) {
                            console.log(`Error preparing statement for activations: ${err.message}`);
                        }
                    });
                    // 3. Per ogni attivazione
                    for (const [f, v] of Object.entries(token.activations)) {
                        console.log(`    Inserting activation for token ID: ${tokenId}, feature: ${v[0]}, value: ${v[1]}`);
                        stmt.run(tokenId, v[0], sampleId, v[1], function(err) {
                            if (err) {
                                console.log(`Error inserting activation for token ID: ${tokenId}, feature: ${v[0]}, sample_id: ${sampleId}, value: ${v[1]} - ${err.message}`);
                            } else {
                                console.log(`    Inserting activation for token ID: ${tokenId}, feature: ${v[0]}, sample_id: ${sampleId}, value: ${v[1]}`);
                            }
                        });
                    }
                    stmt.finalize(function(err) {
                        if (err) {
                            console.error(`Error finalizing statement for activations: ${err.message}`);
                        }
                    });
                }
            }
        });
        db.close();
        console.log('Popolamento del database completato.');
    } catch (err) {
        console.error(`Error executing script: ${err.message}`);
        throw err;
    }
}

const db = new sqlite3.Database(path, (err) => {
    if (err) {
        console.error(err.message);
    } else {
        console.log('Connected to the database.');
    }
});

module.exports = db;