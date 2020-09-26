# Predicting Likelihood of a user E-Signing/Requesting a Loan

Real World Machine Learning projects inspired by the need to create.
Project 05: Predicting the Likelihood of a user E-Signing a Loan Based on Financial History
Industry: Fintech


## Description

Lending companies work by analyzing the financial history of their loan applicants,
and choosing whether or not the applicant is too risky to be offered a loan based on a
risk factor.

An example of this is the credit score system used by the US.
Companies acquire these potential applicants through their website, apps
or advertising campaigns.

We are going to asses the quality or these leads/applicants to determine their risk factors
for our employer.

## Getting Started

The product itself is a loan, but that isn't what we are concerned about.

Our job is to develop a model that can identify 'quality' applicants which are
good candidates for our loaning service.

Definition of quality depends on your requirements(interests of your employer) ie: a user you may consider as a
quality applicant based on your dataset.

In our case study specifications, quality applicants are those who reach
a certain key part of the loan application process.

To be more specific, users that are able to complete the electronic signature
phase of the application process.


TLDR: We try to answer the question: Will this user apply for a loan from us?.

Let's go over the data we will be using this time:

As usual we will discard everything we don't need.

age -> Is the age of the user. Integer
pay_schedule -> Loan payback frequency. Can be weekly, bi-weekly, monthly, semi-monthly. String
home_owner -> Does the customer own their house/property?. Boolean
income -> User's monthly income. Integer
years_employed -> Number of years user has worked current job. Integer
current_address_year -> Number if years user has stayed at current address. Integer
personal_account_m -> Number of months user has had personal account for. Integer
personal_account_y -> Number of years user has had personal account for. Integer
has_debt -> Does user have a debt(based on their credit)?. Boolean
amount_requested -> Amount user requested for loan. Integer
risk_score(s) -> User risk score on application decisions(based on different factors). Decimal(s)
ext_quality_score(s) -> User external quality score(s) coming from external platform, eg: a P2P Marketplace. Decimal(s)
inquiries_last_month -> How many inquiries user has had in the last month. Integer
e_signed? -> Did user e-sign the loan?. This is our response variable. Boolean


### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. [Elias Adjei](https://adjeielias90.github.io)

## Version History


* 0.1
    * Initial Release
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()


## License

This project is licensed under the GNU GPL License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)