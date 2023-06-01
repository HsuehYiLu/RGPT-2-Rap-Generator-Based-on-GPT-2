# RGPT-2: Rap Generator Based on GPT-2

Authors: Kurtis Shen, Hsueh-Yi Lu
Columbian College of Arts & Sciences, George Washington University
Washington, DC 20052
Emails: lootlecoop@gwmail.gwu.edu, hsuehyi_lu2580@gwmail.gwu.edu

## Abstract
This project aims to generate rap verses that are humanly realistic and distinguishable from other genres. The model, RGPT-2, is based on fine-tuning the GPT-2 text generation model. Inspired by the GPoet-2 model, our implementation utilizes a two-model consecutive generating process. The second model is trained on reversed training texts, capturing more structure within a verse. Our model can generate hyper-human-like rap verses without any provided words, easily categorizable into the hip-hop genre.


## Introduction
NLP is a developing technology with many challenges, and this project explores the use of language generation models to train a rapper model capable of creating rap lyrics. By training the model on rap lyrics data and using a classification model, we can identify the generated rap lyrics and distinguish them from other genres.

## Related Work
### Language Generation
Natural Language Generation (NLG) is a software process that produces natural language output. It involves constructing computer systems that can generate understandable texts in English or other human languages from non-linguistic representations of information.

## Datasets
### Source
The dataset used in this project is from the Hip-Hop Encounters Data Science and Music Genre Classification from Kaggle. The dataset includes the names of several rappers and verses of their songs. Music Genre Classification dataset contains lyrics and their related music genre, including rap and pop genres.

### Text Generation
Training and validation sets are retrieved from a public open-source GitHub repository. Each file represents an artist, and songs are separated by two lines, verses by one line, and bars (sentences) stand on their own line.

### Genre Classification
Training and testing sets are retrieved from the music platform Genius.com. The dataset contains lyrics and their related music genre (pop or hip-hop).

## Models
### Inspiration
The RGPT-2 model is based on GPT-2, a text generation model developed by OpenAI. Inspired by the GPoet-2 model, we adopted a two-model consecutive generating process. Our approach captures more structure within a verse by training a second model on reversed training texts.

### Framework
To implement our idea, we trained the standard GPT-2 model on our lyrics dataset. We then trained another GPT-2 model on.
