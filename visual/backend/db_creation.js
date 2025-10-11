'use strict'

const child_process = require('child_process');
const creation = child_process.execSync('type db_init.sql | sqlite3 sae_tokens.db');


