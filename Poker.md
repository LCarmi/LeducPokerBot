Dear students enrolled to the course titled "Economics and Computation",

Due to the restrictions for COVID-19 and the  impossibility to have exams during this period, we decided to change the format of the engaged version of the exam, as follows:

- **midterms**: they will be postponed and, if necessary, grouped in a unique or a pair of midterms;
- **challenge**: see the details below.

 

The goal of the first three challenges is the deployment of a bot for playing a 2-player poker game (we hope to have the forth  challenge in the classroom). Then, the bots of the different teams will  play one against the other in a tournament. More precisely, your software is just required to produce a strategy  for player 1 and a strategy for player 2, and not to play actually the  game. The play phase will be done by the lecturers, who will collect  your strategies, parse them, and subsequently evaluate the expected utility achieved by each bot.

Nowadays, the mainstream approach to solve a 2-player  zero-sum game suggests that a bot should be composed of a number of  components, namely:

- **Abstraction generator**: given the game  tree, a reduced version of the game is produced, even with loss of  information (i.e., the optimal solution for the abstract game has no  guarantee to be an optimal solution for the original, non-abstracted, game); this step is motivated by the fact that huge  games cannot be practically solved exactly.
- **Game solver**: given the abstract game, the optimal strategies of the players are  computed by using an anytime algorithm with theoretical guarantees to  converge eventually to the optimal solution; the strategies produced by  the game solver are said blueprint.
- **Subgame re-solver**: the blueprint is mapped to the original game. In doing this task, the  blueprint is refined by expanding the abstraction and then re-solving a  subgame under the assumption that the behavior of the players off such subgame is that prescribed by the blueprint. The rationale is  that the blueprint may be too coarse when played in practice, and an  on-the-fly refinement may lead to improve the performance of the  strategy dramatically.

In the design of the above components, there are various options that can be adopted in practice, thus leading potentially to a  wide spectrum of different bots. We expect that the teams investigate  different options to choose the best according to their investigation. Most importantly, such an investigation  represents the most challenging task to face from the scientific point  of view.

The teams are free to use the programming language they  prefer. We suggest, for simplicity, Python. However, we will only  evaluate the strategies returned by your software (see below for the  definition of the data structure), and not the code of your software. Furthermore, we will not evaluate the work done for  single challenges, but the evaluation will be done at the end of the  three challenges. In particular, we expect the tournament will be held  during the first week of the forth part of the course, to allow you to have a sufficiently long time to complete the  bot. (Hopefully, the COVID-19 emergency will be terminated by the middle of May.) However, we prefer not to provide the details on all the three components immediately, but we will post them with a given schedule (see below) to force the teams to focus on  specific problems progressively.

The evaluation of the bot will take into account:

- the *deployment* of a bot that actually works;
- the *design choices* done by the teams (we expect that every team will present their bot briefly in the classroom or by Microsoft Teams);
- the *ranking* of the bot in the results of the tournament (the winner will receive incentives in terms of points).

The maximum number of points will be 8 and will be summed to the point of the midterms/exam.

 

The first assignment (corresponding to the first challenge) is the **abstraction generator**. More precisely:

- you are given a data structure for the game tree and  some examples of poker games, provided as text files, distinguishing for different deck size (before the tournament, we will send you the  specific version of the game to use in the tournament);
- you are required to parse the files coding the game tree;
- you are required to produce an abstracted version of the game tree,  together with the correspondence between the actions of the two  (original and abstract) trees;
- you are required to  produce some function mapping the strategies of the abstract game to the strategy of the original game (without any strategy refinement).

The second assignment will be the **game solver** and its details will be published by 3 weeks, while the third assignment will be the **subgame re-solver** and its details will be published by 6 weeks.

 

To support your work, we produced a video lecture providing some  groundings on the generation of abstractions. Furthermore, next week the lecture of Thursday will be devoted to the clarification of possible  doubts on abstraction generation. The link to the video is (https://youtu.be/VgnTQtC2x18), while the link to slides is  (https://www.dropbox.com/s/urp0w0kzfakaecy/Abstraction%20Generation.pdf?dl=0). Here  (https://www.dropbox.com/sh/qmd87grb3qnnr3r/AAB_cgr6wzK05D7N_fY_rA1Wa?dl=0), instead, you can find:

- examples of Kuhn poker and of Leduc poker,
- an example of strategy for Kuhn poker,
- the description of the format of the input files and the description of the output files describing the strategies that your software is required  to produce.

Best regards,

Nicola.