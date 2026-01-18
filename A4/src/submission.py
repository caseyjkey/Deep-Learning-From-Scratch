from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent
# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.getDirection() gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    legalMoves = gameState.getLegalActions()

    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)

    # BEGIN_HIDE
    # END_HIDE

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_HIDE
    # END_HIDE
    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    def isEnd(state, agent):
      return state.isWin() or state.isLose() or not state.getLegalActions(agent)
    
    def vMinimax(state, depth, agent):
      if isEnd(state, agent):
        return state.getScore()
      
      if depth == 0:
        return self.evaluationFunction(state)
      
      actions = state.getLegalActions(agent)
      
      agents = state.getNumAgents()
      nextAgent = (agent + 1) % agents
      
      if nextAgent == 0:
        nextDepth = depth - 1
      else:
        nextDepth = depth
      
      if agent == 0:
        maxVal = float('-inf')
        for action in actions:
          successor = state.generateSuccessor(agent, action)
          value = vMinimax(successor, nextDepth, nextAgent)
          if value > maxVal:
            maxVal = value
        return maxVal
      else:
        minVal = float('inf')
        for action in actions:
          successor = state.generateSuccessor(agent, action)
          value = vMinimax(successor, nextDepth, nextAgent)
          if value < minVal:
            minVal = value
        return minVal
    
    actions = gameState.getLegalActions(0)
    
    if not actions:
      return None
    
    bestAction = None
    bestVal = float('-inf')
    
    for action in actions:
      successor = gameState.generateSuccessor(0, action)
      value = vMinimax(successor, self.depth, 1)
      if value > bestVal:
        bestVal = value
        bestAction = action
    
    return bestAction

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    def isEnd(state, agent):
      return state.isWin() or state.isLose() or not state.getLegalActions(agent)
    
    def vMinimax(state, depth, agent, alpha, beta):
      if isEnd(state, agent):
        return state.getScore()
      
      if depth == 0:
        return self.evaluationFunction(state)
      
      actions = state.getLegalActions(agent)
      
      agents = state.getNumAgents()
      nextAgent = (agent + 1) % agents
      
      if nextAgent == 0:
        nextDepth = depth - 1
      else:
        nextDepth = depth
      
      if agent == 0:
        maxVal = float('-inf')
        for action in actions:
          successor = state.generateSuccessor(agent, action)
          value = vMinimax(successor, nextDepth, nextAgent, alpha, beta)
          if value > maxVal:
            maxVal = value
          alpha = max(alpha, maxVal)
          if beta <= alpha:
            break
        return maxVal
      else:
        minVal = float('inf')
        for action in actions:
          successor = state.generateSuccessor(agent, action)
          value = vMinimax(successor, nextDepth, nextAgent, alpha, beta)
          if value < minVal:
            minVal = value
          beta = min(beta, minVal)
          if beta <= alpha:
            break
        return minVal
    
    actions = gameState.getLegalActions(0)
    
    if not actions:
      return None
    
    bestAction = None
    bestVal = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for action in actions:
      successor = gameState.generateSuccessor(0, action)
      value = vMinimax(successor, self.depth, 1, alpha, beta)
      if value > bestVal:
        bestVal = value
        bestAction = action
      alpha = max(alpha, bestVal)
    
    return bestAction

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    def isEnd(state, agent):
      return state.isWin() or state.isLose() or not state.getLegalActions(agent)
    
    def vExpectimax(state, depth, agent):
      if isEnd(state, agent):
        return state.getScore()
      
      if depth == 0:
        return self.evaluationFunction(state)
      
      actions = state.getLegalActions(agent)
      
      agents = state.getNumAgents()
      nextAgent = (agent + 1) % agents
      
      if nextAgent == 0:
        nextDepth = depth - 1
      else:
        nextDepth = depth
      
      if agent == 0:
        maxVal = float('-inf')
        for action in actions:
          successor = state.generateSuccessor(agent, action)
          value = vExpectimax(successor, nextDepth, nextAgent)
          if value > maxVal:
            maxVal = value
        return maxVal
      else:
        total = 0
        for action in actions:
          successor = state.generateSuccessor(agent, action)
          value = vExpectimax(successor, nextDepth, nextAgent)
          total += value
        return total / len(actions)
    
    actions = gameState.getLegalActions(0)
    
    if not actions:
      return None
    
    bestAction = None
    bestVal = float('-inf')
    
    for action in actions:
      successor = gameState.generateSuccessor(0, action)
      value = vExpectimax(successor, self.depth, 1)
      if value > bestVal:
        bestVal = value
        bestAction = action
    
    return bestAction

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  score = currentGameState.getScore()
  pos = currentGameState.getPacmanPosition()
  food = currentGameState.getFood().asList()
  capsules = currentGameState.getCapsules()
  ghosts = currentGameState.getGhostStates()
  
  if food:
    foodDists = [manhattanDistance(pos, f) for f in food]
    minFood = min(foodDists)
    foodScore = 15.0 / (minFood + 1)
    nearbyFood = sum(1 for d in foodDists if d < 4)
    foodScore += nearbyFood * 2.0
  else:
    foodScore = 0
  
  ghostPos = [g.getPosition() for g in ghosts]
  scared = [g for g in ghosts if g.scaredTimer > 0]
  
  if ghostPos:
    minGhost = min([manhattanDistance(pos, gp) for gp in ghostPos])
  else:
    minGhost = float('inf')
  
  capsuleScore = 0
  for cap in capsules:
    capDist = manhattanDistance(pos, cap)
    
    nearbyGhosts = sum(1 for gp in ghostPos if manhattanDistance(gp, cap) < 6)
    
    if nearbyGhosts > 0:
      if minGhost < 5:
        capsuleScore += 50.0 / (capDist + 1)
      else:
        capsuleScore += 2.0 / (capDist + 1)
    else:
      capsuleScore += 1.5 / (capDist + 1)
  
  ghostPenalty = 0
  capsuleLuringBonus = 0
  
  for g in ghosts:
    if g.scaredTimer <= 0:
      gDist = manhattanDistance(pos, g.getPosition())
      
      if gDist < 3:
        basePenalty = 20.0 / (gDist + 1)
        
        ghostDir = g.getDirection()
        if ghostDir:
          gx, gy = g.getPosition()
          px, py = pos
          
          dirVector = Actions.directionToVector(ghostDir)
          toPacmanVector = ((px - gx) / max(gDist, 1), (py - gy) / max(gDist, 1))
          
          alignment = dirVector[0] * toPacmanVector[0] + dirVector[1] * toPacmanVector[1]
          
          if alignment > 0.4:
            basePenalty *= 1.3
          elif alignment < -0.4:
            basePenalty *= 0.8
        
        ghostPenalty -= basePenalty
      elif gDist < 5:
        ghostPenalty -= 2.0 / (gDist + 1)
  
  if capsules:
    minCapsuleDist = min([manhattanDistance(pos, cap) for cap in capsules])
    if minCapsuleDist < 4:
      for g in ghosts:
        if g.scaredTimer <= 0:
          gDist = manhattanDistance(pos, g.getPosition())
          if gDist < 6:
            ghostDir = g.getDirection()
            if ghostDir:
              gx, gy = g.getPosition()
              px, py = pos
              
              dirVector = Actions.directionToVector(ghostDir)
              toPacmanVector = ((px - gx) / max(gDist, 1), (py - gy) / max(gDist, 1))
              
              alignment = dirVector[0] * toPacmanVector[0] + dirVector[1] * toPacmanVector[1]
              
              if alignment > 0.5:
                lureBonus = 8.0 / (gDist + 1) / (minCapsuleDist + 1)
                capsuleLuringBonus += lureBonus
  
  scaredBonus = 0
  for s in scared:
    sDist = manhattanDistance(pos, s.getPosition())
    ghostDir = s.getDirection()
    directionBonus = 1.0
    
    if ghostDir and sDist < 8:
      gx, gy = s.getPosition()
      px, py = pos
      
      dirVector = Actions.directionToVector(ghostDir)
      toPacmanVector = ((px - gx) / max(sDist, 1), (py - gy) / max(sDist, 1))
      
      alignment = dirVector[0] * toPacmanVector[0] + dirVector[1] * toPacmanVector[1]
      
      if alignment > 0.5:
        directionBonus = 1.5
      elif alignment < -0.5:
        directionBonus = 0.7
    
    scaredBonus += (100.0 * directionBonus) / (sDist + 1)
  
  return score + foodScore + capsuleScore + ghostPenalty + scaredBonus + capsuleLuringBonus

# Abbreviation
better = betterEvaluationFunction
