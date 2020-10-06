# Project of Economics and Computation - 2019/2020 

<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="header" />
</p>
<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

### This repository contains the software (and other supporting files for the development) required for the Economics and Computation course.

### The components of the team are : 
- ### Luca Carminati 
- ### Teresa Costa Cañones 
- ### Robert Stefano Chinga Barazorda ([@robertsteven97](https://github.com/robertsteven97))

## For further details look at the [presentation](https://github.com/LCarmi/LeducPokerBot/blob/master/Final%20Presentation/Final%20Presentation.pdf)

## Structure of the Repository
- [manager.py](https://github.com/LCarmi/LeducPokerBot/blob/master/manager.py) contains the code needed to run the learning algortihm of the bot, and the code related to refinements of the strategy
- [game.py](https://github.com/LCarmi/LeducPokerBot/blob/master/game.py) contains the main utilities needed to manage a single instance of a game, such as abstraction generators and exploitability evaluators
- [node.py](https://github.com/LCarmi/LeducPokerBot/blob/master/node.py) contains the basic node data structures, and the recursive versions of optimizing algorithms

### Note: for better performances, we suggest to use [pypy](https://www.pypy.org/) as Python interpreter
