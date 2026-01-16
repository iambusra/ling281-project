# Race & NLP Project

This repository contains data collection, annotation, and analysis code for an NLP project investigating race-, nationality-, and migration-related discourse in Turkish online forums.

## Project Overview

The project focuses on user-generated content from **Ekşi Sözlük**, targeting discussions related to migration, nationality, and ethnicity. Titles containing specific keywords were scraped and subsequently annotated for **sentiment** and **topic** using an LLM-based annotation pipeline.

The goal is to enable downstream linguistic and computational analysis of how different groups are discussed in Turkish online discourse.

## Data Collection

Data was scraped from Ekşi Sözlük using the open-source scraper:

- **erenseymen / eksisozluk-scraper**

Scraper scripts used in this project are located in the `scraper-scripts/` directory.

### Keyword Selection

Titles containing the following keywords were collected:

- `suriye`, `suriyeli`
- `rus`, `rusya`
- `ukrayna`, `ukraynalı`
- `göçmen`
- `sığınmacı`
- `mülteci`

These keywords were chosen to capture discussions related to migration, refugees, and national/ethnic groups.

## Annotation

The scraped data was annotated for:

- **Sentiment**
- **Topic**

Annotation was performed using **GPT-5 Nano**.

- Annotation scripts are located in: `annotation-scripts/`
- Annotated datasets are stored in: `annotated-data/`

The annotation pipeline is fully reproducible given access to the model and API credentials.

## Analysis

Basic statistical and exploratory analyses are implemented in the `basic-analysis/` directory.  
These scripts include:

- Descriptive statistics
- Distributional analyses of sentiment and topics
- Preliminary comparisons across keyword groups

Further modeling and inferential analyses can be built on top of this structure.

## Repository Structure

```text
.
├── scraper-scripts/       # Ekşi Sözlük scraping scripts
├── annotation-scripts/    # GPT-based annotation pipeline
├── annotated-data/        # Annotated datasets
├── basic-analysis/        # Exploratory and statistical analysis scripts
└── README.md
