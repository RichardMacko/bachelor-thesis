import json
from instagrapi import Client

class InteractionManager:
    def __init__(self, client):
        self.client = client

    def get_following_dict(self, username):
        following_dict = {}
        uid = self.client.user_id_from_username(username)
        following = self.client.user_following(uid)
        following_dict[username] = []

        for user in following.values():
            following_dict[user.username] = []

        return following_dict

    def update_relationships_in_dict(self, following_dict):
        with open("json_data/following_dictionary.json", "w", encoding="utf8") as backup_file:
            num_of_followings = len(following_dict.keys() - 1)
            for user in following_dict.keys():
                uid = self.client.user_id_from_username(user)
                f = self.client.user_following(uid, True, num_of_followings)
                for followed in f.values():
                    if followed.username in following_dict:
                        following_dict[followed.username].append(user)

            json.dump(following_dict, backup_file, indent=4)
        return following_dict

    def getRelationshipsFromUsersFollowings(self, username):
        following_dict = self.get_following_dict(username)
        following_dict = self.update_relationships_in_dict(following_dict)
        return following_dict

    def get_interactions(self, user_a, user_b):
        interactions = 0
        posts_a = self.get_posts(user_a)
        for post in posts_a:
            likes = self.get_likes(post.id)
            comments = self.get_comments(post.id)

            if user_b in likes:
                print(user_b + " likes a photo of user: " + user_a)
                interactions += 1
            if user_b in comments:
                interactions += 1
                print(user_b + " commented a photo of user: " + user_a)

        posts_b = self.get_posts(user_b)
        for post in posts_b:
            likes = self.get_likes(post.id)
            comments = self.get_comments(post.id)

            if user_a in likes:
                interactions += 1
                print(user_a + " likes a photo of user: " + user_b)
            if user_a in comments:
                interactions += 1
                print(user_a + " commented a photo of user: " + user_b)

        return interactions

    def get_posts(self, user):
        print("getting posts from: " + user)
        try:
            uid = self.client.user_id_from_username(user)
            medias = self.client.user_medias(uid, 5)
            return medias
        except:
            print(f" Account {user} is private.")
            return []

    def get_likes(self, post_id):
        likes = self.client.media_likers(post_id)
        listOfLikers = []
        for username in likes:
            listOfLikers.append(username.username)
        return listOfLikers

    def get_comments(self, post_id):
        comments = self.client.media_comments(post_id)
        listOfCommenters = []
        for comment in comments:
            listOfCommenters.append(comment.user.username)
        return listOfCommenters

    def process_interactions(self, following_dictionary, interaction_dict, reverse_interactions):
        for userA, users in following_dictionary.items():
            for userB in users:
                if (userA, userB) in interaction_dict.keys():
                    print(f"{userA} a {userB} were searched, continuing")
                    pass
                # only used if program is running multiple times
                elif (userA, userB) in reverse_interactions:
                    print(f"{userA} a {userB} were searched, continuing")
                    pass
                else:
                    if (userB, userA) in interaction_dict.keys():
                        interaction_dict[(userB, userA)] += 1
                        reverse_interactions.append((userA, userB))
                        print("REVERSE: " + userA + " " + userB)
                        print(reverse_interactions)
                    else:
                        numberOfInteractions = self.get_interactions(userA, userB)
                        interaction_dict[(userA, userB)] = 0
                        interaction_dict[(userA, userB)] += numberOfInteractions + 1
                    print(interaction_dict)

        with open("json_data/interaction_dict.json", "w", encoding="utf8") as f:
            json.dump({f"{k[0]}|{k[1]}": v for k, v in interaction_dict.items()}, f, indent=4)

        with open("json_data/reverse_interactions.json", "w", encoding="utf8") as f:
            json.dump(reverse_interactions, f, indent=4)

        return interaction_dict
