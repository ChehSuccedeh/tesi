"use strict";

const cors = require("cors");
const express = require("express");
const app = express();
app.use(cors());
app.use(express.json());
const morgan = require("morgan");



const SERVER_URL = "http://localhost";
const SERVER_PORT = 3001;

app.use(morgan("dev"));
const db_interface = require("./db_interface");

app.get("/api/features/:featureNumber", async (req, res) => {
    const featureNumber = req.params.featureNumber;
    try {
        const values = await db_interface.getTokensAndActivationsByFeature(featureNumber);
        res.json(values);
    } catch (err) {
        res.status(500).json({ error: "Errore nel recupero delle feature" });
    }
});

app.get("/api/featureNumber", async (req, res) => {
    try {
        const values = await db_interface.getMaxFeatureNumber();
        res.json(values);
    } catch (err) {
        res.status(500).json({ error: "Errore nel recupero del numero massimo di feature" });
    }
});

app.listen(SERVER_PORT, () => {
    // try{
    //     db_interface
    // } catch(err){
    //     console.log(err);
        
    // }
    console.log(`Server in ascolto sulla porta ${SERVER_PORT}`);
    console.log(`URL: ${SERVER_URL}:${SERVER_PORT}`);
});