# Directing Customers to Subscriptions using app activity

Real World Machine Learning projects inspired by the need to create.
Project 03: Directing Customers to Subscriptions/Suggestions Through App Behavior Analysis
Industry: Fintech


## Description

Evaluating users to soecify whether or noot they are likely to subscribe to
a paid service, based on the behavior of such user on the platform.

Service must have both a free and paid service, and this model will mostly
target the free users.

The company that we work for is a large Fintech firm that offers a subscription based
paid service with has a free version as well. This app or website allows users to
track all thier finances in one place.
In the free version some of the features are restricted.
You know the rest.



## Getting Started

Our job is to build a model that can correctly identify users
that are not likely to purchase the premium, so additional offers can be added to their free subscription.
By doing this we can also identify users that are more likely to purchase extra features
so we can target them with personalized ads.

We have a very specific set of data from our app, including user app
behaviour. We can use this data in our model.
The app usage behaviour is only for the first 24 hours free trial of the premium
features.
This very specific data is from the first day the user registered on the app.

That out of the way, let's go over the data we'll be using.
We will discard everything else.

user -> uid of the user.
first_open -> datetime when user first signed up.
day_of_week -> numerical respresentation of the day of the week user enrolled, [0-6 -> Sunday - Saturday].
hour -> hour if day user enrolled.
age -> users age.

screen_list -> all screens in our app that the user has visited on the first 24 hours in the app(We'll use this a lot).
numscreens -> number of screens user visited
minigame -> user played mini game. value is a boolean.
liked -> did the user like any of our features? also boolean
used_premium_feature -> if the user used any premium feature while on trial. boolean.
enrolled -> whether or not they enrolled to use our premium service after the trial. boolean.
enrolled_date -> date user enrolled.

To Recap: Our aim is to produce a model that will label every new user
as likely or unlikely to subscribe to a paid service.
A company armed with this information can better narrow down targeted marketing to users.


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