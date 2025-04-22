from instagrapi import Client
from session_manager import SessionManager
from exception_handler import ExceptionHandler
from Interaction_manager import InteractionManager
from graph_builder import GraphVisualizer

client = Client()
session_manager = SessionManager(client)
interaction_manager = InteractionManager(client)
exception_handler = ExceptionHandler(client)
client.handle_exception = exception_handler.handle_exception

name = input("Type username: ")
password = input("Type password: ")
client.login(name, password)
session_manager.load_session()

username = input("Enter the username to get interactions for: ")
following_dict = interaction_manager.getRelationshipsFromUsersFollowings(username)

interaction_dict = {}
reverse_interactions = []
# USE IT IN CASE OF RUNNING THE SCRIPT NOT FOR THE FIRST TIME
#with open("json_data/following_dictionary.json", "r", encoding="utf8") as t:
#    following_dict = json.load(t)
#with open("json_data/interaction_dict.json", "r", encoding="utf8") as t:
#    raw_data = json.load(t)
#    interaction_dict = {tuple(k.split('|')): v for k, v in raw_data.items()}
# with open("json_data/reverse_interactions.json", "r", encoding="utf8") as t:
#     reverse_interactions = json.load(t)

interaction_dict = interaction_manager.process_interactions(following_dict, interaction_dict, reverse_interactions)

graph_visualizer = GraphVisualizer(following_dict, interaction_dict)
graph_visualizer.build_graph()
graph_visualizer.show_density()



