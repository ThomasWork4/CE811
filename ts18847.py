import random
from player import Bot
from game import State


class myBot(Bot):
    
    def onGameRevealed(self, players, spies):
        # Instantiate all the dictionaries required to store the decisional
        # Data each round 
        self.spies = spies
        self.SuspectedSpyScore = {}
        self.DefiniteSpyScore = {}
        self.failed_missions_voted_up = {}
        self.failed_missions_been_on = {}
        self.teams_created_without_self = {}
        self.successful_team = {}
        self.playerID = {}
        self.First_game_fail = ""
        ID_Counter = 0
        # Make sure there is a key and starting value for each player in
        # The game as well as a player ID 
        for p in players:
            self.failed_missions_voted_up[p] = 0
            self.failed_missions_been_on[p] = 0
            self.SuspectedSpyScore[p] = 0
            self.DefiniteSpyScore[p] = 0
            self.teams_created_without_self[p] = 0
            self.playerID[p] = str(ID_Counter)
            ID_Counter += 1
        pass


    def select(self, players, count):

        # If i am a spy, create a random team from all the players that don't
        # Include the other spy
    
        if self.spy:
            team = []
            for p in self.others():
                if p not in self.spies:
                    team.append(p)
                else:
                    continue
            return [self] + random.sample(team, count - 1)




        # From the games list of successful teams, choose the team that has 
        # Succeeded the most times, with me in the team
        
        if not self.spy:
            # Initialise variables and sorted the list of suspicion scores and successful teams
            Most_Successful_Team = ""
            First_team_that_failed = ""
            Final_team = []
            Dont_Pick = []
            HighestLikelySpy = sorted(self.SuspectedSpyScore.items(), key=lambda x: x[1])
            successful_teams = sorted(self.successful_team.items(), key=lambda x: x[1], reverse=True)
            Teams_with_me_in = 0

            # If its the first turn of the game, or the first turn was sabotaged and there is no successes yet
            if self.game.turn == 1:
                return [self] + random.sample(self.others(), count - 1)

            # If the first game lost or there hasn't been a successful team to choose from
            # Pick yourself and anyone who was not in the last game
            # If not enough players to choose from, pick the least suspected spy
            if len(successful_teams) == 0: 
                Final_team.append(self)
                First_team_that_failed = self.First_game_fail
                for ID in First_team_that_failed:
                    for k, v in self.playerID.items():
                        if v == ID:
                            Dont_Pick.append(k)
                for x in range(count - 1):
                    if self.game.players[x] not in Dont_Pick and self.game.players[x] != self:
                        Final_team.append(self.game.players[x])
                if len(Final_team) != count:
                    for k, v in HighestLikelySpy:
                        if k != self and k not in Final_team:
                            Final_team.append(k)
                            if len(Final_team) == count:
                                break
                return Final_team
                        

            # In the order of the ordered list of successful teams, if the team contains me, use that team as a baseline
            for team in successful_teams:
                if self.playerID[self] in team[0]:
                    Most_Successful_Team = team[0]
                    Teams_with_me_in += 1
                    break
                else:
                    continue

            # If there are no teams with me in, just use the most successful team
            if Teams_with_me_in == 0:
                Most_Successful_Team = successful_teams[0][0]


            # For each PlayerID in that team, add this player to the final team
            for ID in Most_Successful_Team:
                for k, v in self.playerID.items():
                    if v == ID:
                        Final_team.append(k)


            # If the length of the team is greater than we need
            # remove the first person that isn't me 
            if len(Final_team) > count:
                for x in Final_team:
                    if x != self:
                        Final_team.remove(x)
                        break
                    else:
                        continue

            # If the length of the team is less than we need
            # For each player in the most likely spies list
            # If the player isn't me, and the player is not already in the list
            # Add them to the team until we have the team count we require
            if len(Final_team) < count:
                for k, v in HighestLikelySpy:
                    if k != self and k not in Final_team:
                        Final_team.append(k)
                        if len(Final_team) == count:
                            break
                
            return Final_team 

    def onTeamSelected(self, leader, team):
        if leader not in team:
            self.teams_created_without_self[leader] += 1
        pass

    def vote(self, team):
        # If i'm a spy, if there are no spies in the team then vote against it
        # Otherwise, vote for it 
        if self.spy:
            SpyCount = 0
            for p in self.game.team:
                if p in self.spies:
                    SpyCount += 1
            if SpyCount == 0:
                return False
            else:
                return True 

        
        # If i'm not a spy, if its the last mission, vote for it, otherwise only vote for a team
        # If it doesn't contain any definite spies, and it doesn't cointain the highest
        # 2 suspected spies
        if not self.spy:
            HighestLikelySpy = sorted(self.SuspectedSpyScore.items(), key=lambda x: x[1], reverse=True)
            for p in team:
                if self.game.tries == 5:
                    return True
                elif self.DefiniteSpyScore[p] == 1 or p in HighestLikelySpy[:2] :
                    return False
                else:
                    return True

    def sabotage(self):
        # If there is only one spy in the game and it's not the first turn then sabotage
        # If there is 2 spies but I'm the leader of the team, then sabotage
        # Otherwise don't sabotage
        SpyCount = 0
        for p in self.game.team:
            if p in self.spies:
                SpyCount += 1
        if SpyCount < 2 and self.game.turn > 1:
            return True
        if SpyCount > 2 and self == self.game.leader:
            return True
        else:
            return False

    def onMissionComplete(self, num_sabotages):
        Successful_Team_Combination = ""
        First_Team_fail = ""

        # If there was 2 sabotages for 2 team members, we know who our spies are 
        if len(self.game.team) == 2 and num_sabotages == 2:
            for p in self.game.team:
                self.DefiniteSpyScore[p] += 1

        # If the game was sabotaged, increment the relevant dictionary values by one depending
        # On which players are currently in the team
        if num_sabotages > 0:
            for i in self.game.team:
                self.failed_missions_been_on[i] += 1
            for x in range(len(self.game.votes)):
                p = self.game.players[x]
                if self.game.votes[x] == True:
                    self.failed_missions_voted_up[p] += 1

        # If the first mission failed, then take not of the team that failed            
        if num_sabotages > 0 and self.game.turn == 1:
              for p in self.game.team:
                    self.First_game_fail += self.playerID[p]

        # For each player in the game depending on what decisions they made this round,
        # Calculate their SuspectedSpyScore 
        for p in self.game.players:
            if p == self:
                pass
            else:
                SuspectScore = (self.failed_missions_voted_up[p] + self.failed_missions_been_on[p] + self.teams_created_without_self[p]) / 3
                self.SuspectedSpyScore[p] += SuspectScore

        # If the game wasn't sabotaged, then note the team and it's players and increment their score
        # by 1 if they already in the list, otherwise set it to one
        if num_sabotages == 0:
            for p in self.game.team:
                Successful_Team_Combination += self.playerID[p]
            if Successful_Team_Combination in self.successful_team:
                self.successful_team[Successful_Team_Combination] += 1
            else:
                self.successful_team[Successful_Team_Combination] = 1
        return {}
