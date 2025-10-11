'use strict';

const DB = require('./db');

/**
 * Restituisce per ogni sample una lista di oggetti { token_str, feature_value }
 * @param {string} feature - Il nome della feature da estrarre
 * @returns {Promise<Array<{ sample_id: any, tokens: Array<{ token_str: string, feature_value: any }> }>>}
 */
function getTokensAndActivationsByFeature(db, feature_id, callback) {
  const query = `
    SELECT s.id AS sample_id, t.id AS token_id, t.token_text,
           COALESCE(a.value, 0) AS value
    FROM samples s
    JOIN tokens t ON t.sample_id = s.id
    LEFT JOIN activations a ON a.token_id = t.id AND a.feature_id = ?
    ORDER BY s.id, t.id
  `;
  db.all(query, [feature_id], (err, rows) => {
    if (err) return callback(err);

    // Raggruppa per sample
    const result = {};
    for (const row of rows) {
      if (!result[row.sample_id]) result[row.sample_id] = [];
      result[row.sample_id].push({
        token_id: row.token_id,
        token_text: row.token_text,
        value: row.value
      });
    }
    callback(null, result);
  });
}

function getMaxFeatureNumber(db, callback) {
  const query = `SELECT MAX(feature_id) AS max_feature FROM activations`;
  db.get(query, [], (err, row) => {
    if (err) return callback(err);
    callback(null, row.max_feature);
  });
}

module.exports = {
    getTokensAndActivationsByFeature,
    getMaxFeatureNumber
};