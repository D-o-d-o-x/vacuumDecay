# Project vacuumDecay

Project vacuumDecay is a framework for building AIs for games.  
Avaible architectures are
 - those used in Deep Blue (mini-max / expecti-max)
 - advanced expecti-max exploration based on utility heuristics
 - those used in AlphaGo Zero (knowledge distilation using neural-networks)

A new AI is created by subclassing the State-class and defining the following functionality (mycelia.py provies a template):
 - initialization (generating the gameboard or similar)
 - getting avaible actions for the current situation (returns an Action-object, which can be subclassed to add additional functionality)
 - applying an action (the state itself should be immutable, a new state should be returned)
 - checking for a winning-condition (should return None if game has not yet ended)
 - (optional) a getter for a string-representation of the current state
 - (optional) a heuristic for the winning-condition (greatly improves capability)
 - (optional) a getter for a tensor that describes the current game state (required for knowledge distilation)
 - (optional) interface to allow a human to select an action

### Current state of the project
The only thing that currently works is the AI for Ultimate TicTacToe.  
It uses a trained neural heuristic (neuristic)  
You can train it or play against it (will also train it) using 'python ultimatetictactoe.py'
