# From Data to Translation: Leveraging AI for Efficient and Accurate Translation of Scientific Reports 

# Phase 3: Rule-Based Preferential Translations

## Description

This phase implements a rule-based find-and-replace algorithm designed to work in conjunction with the fine-tuned translation models from Phase 2. The system preserves critical scientific terminology, location names, species names, and acronyms that require consistent translation across CSAS documents, ensuring accuracy and maintaining standardised terminology.

A significant challenge emerged in preventing translation models from modifying the replacement tokens themselves. Through iterative testing and refinement, the approach evolved from complex tokenisation schemes to a more robust word-based system that maintains translation integrity while preserving essential scientific terminology.

## Key Components

### Rule-Based Translation System
- Find-and-replace algorithm for preserving critical terminology
- Key-value pair management for locations, species, acronyms, and scientific terms
- Error detection and fallback mechanisms
- Integration layer with fine-tuned translation models

### Token Design Evolution
The token design underwent several iterations to prevent accidental translation:

1. **Initial Approach**: Complex tokens like `__TECHNICAL_001__` proved problematic as models would translate components or remove special characters
2. **Special Character Tokens**: Even unique characters like lozenges were affected by translation models
3. **Encoder Integration**: Adding tokens directly to model encoders degraded translation quality
4. **Final Solution**: Simple word-based tokens (`SITE0013`, `ACRONYM0001`, `NOMENCLATURE0008`, `TAXON0045`) chosen for language-neutral spelling

### Additional Fine-Tuning
An additional layer of finetuning was added to the translation models to train them to properly handle translation tokens. This fine-tuning had the added benefit of training the models to treat the translation tokens with the correct context, and as the correct part-of-speech.

## Challenges Addressed

1. **Token Preservation**: Preventing translation models from modifying replacement tokens during translation
2. **Model Integration**: Balancing rule-based interventions with neural translation quality
3. **Error Handling**: Ensuring all tokens are properly accounted for before replacement
4. **Terminology Consistency**: Maintaining standardised translations for scientific terms across documents

## Technical Solutions

### Token Selection Strategy
- **SITE**: Identical spelling in English and French
- **NOMENCLATURE**: Identical spelling in both languages  
- **TAXON**: Identical spelling in both languages
- **ACRONYM**: Commonly borrowed word in French, `ACRONYM0001` is unlikely to be translated to `ACRONYME0001`

### Error Prevention
- Pre-replacement validation ensures all tokens are accounted for
- Automatic fallback to standard translation if token integrity is compromised

### Fine-Tuning
- This fine-tuning was used to train models to properly handle translation tokens
- Models were trained to treat the translation tokens with the correct context, and as the correct part-of-speech.

## Outcomes

The rule-based system successfully preserves critical scientific terminology while maintaining translation quality. Additional fine-tuning on token-aware datasets reduced translation errors by approximately 50%, though some edge cases remain. The system provides a robust foundation for consistent, accurate scientific translation.

## All Phases

- **Phase 1**: [Data Gathering and Transformation](https://github.com/KevinCarr42/AI-Translation) (complete)
- **Phase 2**: [AI Translation Fine-Tuning](https://github.com/KevinCarr42/Translation-Fine-Tuning) (complete)
- **Phase 3**: Rule-Based Preferential Translations (complete)
- **Phase 4**: [AI Translation Quality Survey App](https://github.com/KevinCarr42/translation-quality-survey-app) (complete)
- **Phase 5**: [Final AI Translation Model and Translation Quality Evaluation](https://github.com/KevinCarr42/CSAS-Translations) (in-progress)
- **Phase 6**: Deploy the Final Model (in-progress)
