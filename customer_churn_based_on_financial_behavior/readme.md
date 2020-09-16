# Directing Customers to Subscriptions using app activity

Real World Machine Learning projects inspired by the need to create.
Project 04: Minimizing Churn Rate Through Analysis of Financial Habits.
Industry: Fintech


## Description

Evaluating users to specify whether or not they are likely to churn(subscription cancellations).
Our product provides users with a lot of value based services, but most of these services
are based in financial tracking.

The company that we work for is a large Fintech firm that offers a subscription based
value based services.

Our product comes in the form of a subscription which allows users to also manage bank
accounts.

Once again, our employer has been graceful enough to not only include the data we need
but also metadata as well.

Let's go over the data we'll be using.
We will discard everything else.

userid -> MongoDB userid.
churn  -> Active = No | Suspended < 30 = No Else Churn = Yes.
age -> Age of the customer.
city -> City of the customer.
state -> State where the customer lives.
postal_code -> Zip code of the customer.
zodiac_sign -> Zodiac sign of the customer.
rent_or_own -> Does the customer rents or owns a house?.
more_than_one_mobile_device -> Does the customer use more than one mobile device?.
payFreq -> Pay frequency of the cusomter.
in_collections -> Is the customer in collections?.
loan_pending -> Is the loan pending?.
withdrawn_application -> Has the customer withdrawn the loan applicaiton?.
paid_off_loan -> Has the customer paid of the loan?.
did_not_accept_funding -> Customer did not accept funding.
cash_back_engagement -> Sum of cash back dollars received by a customer / No of days in the app.
cash_back_amount -> Sum of cash back dollars received by a customer.
used_ios -> Has the user used an iphone?
used_android -> Has the user used a android based phone?.
has_used_mobile_and_web -> Has the user used mobile and web platforms?.
has_used_web -> Has the user used our Web app?.
has_used_mobile -> Has the user used our mobile app?.
has_reffered -> Has the user referred?.
cards_clicked -> How many times a user has clicked the cards?.
cards_not_helpful -> How helpful was the cards?.
cards_helpful -> How helpful was the cards?.
cards_viewed -> How many times a user viewed the cards?.
cards_share -> How many times a user shared his cards?.
trivia_view_results -> How many times a user viewed trivia results?.
trivia_view_unlocked -> How many times a user viewed trivia view unlocked screen?.
trivia_view_locked -> How many times a user viewed trivia view locked screen?.
trivia_shared_results -> How many times a user shared trivia results?.
trivia_played -> How many times a user played trivia?.
re_linked_account -> Has the user re-linked account?.
un_linked_account -> Has the user unlinked account?.
credit_score -> Customer's credit score.



## Getting Started

Our job is to build a model that can correctly identify users
that are not likely to purchase the premium(churn), so that the company can better engage
those users.

We have been tasked to find users that are most likely to cancel their subscriptions

That out of the way, let's go over the data we'll be using.
We will discard everything else.


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