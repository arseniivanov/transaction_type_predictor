# Transaction type predictor

This repo is meant as a base to predict transaction type using boosted trees and tabular data.

## Description

Somehow, in 2024, traditional banks still have not got around to making personal finance easier by providing tags on transactions for users. The purpose of this project is to train a boosted gradient tree on the data provided by my bank. 

## Technical description

I labelled 1 year of transaction data for this project with labels:

```
label_mapping = {
    'ent': 'Entertainment, alcohol',
    'food': 'Groceries/Food',
    'hobby': 'Hobbies and Home',
    'loan': 'Loan',
    'out': 'Eating out',
    'rent': 'Rent',
    'sub': 'Subscriptions',
    'trans': 'Transport',
    'trip': 'Travel'
}
```

The meaningful parameters for me are:\
Date of transaction\
Transaction amount\
Embedded text description of transaction partner\

### Performance and takeaways

The model does fairly well with a 75-80% weighted accuracy. It does significantly better on common classes, such as food or hobbies. Above 30 samples for a class seems to be crucial for an acceptable accuracy.