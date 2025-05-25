from victus_analyzer import VictusInterpreter

class VictusChat:
    def __init__(self):
        self.interpreter = VictusInterpreter()
        self.interpreter.auto_run = True
        self.interpreter.conversation_history = True
        
    def start_chat(self):
        print("\n=== 🤖 VICTUS Chat ===")
        print("Je suis votre assistant VICTUS. Comment puis-je vous aider ?")
        print("(tapez 'quit' pour quitter)\n")
        
        while True:
            user_input = input("\n👤 Vous : ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Au revoir !")
                break
            
            # Utiliser l'interpréteur pour analyser et répondre
            response = self.interpreter.chat(user_input)
            print(f"\n🤖 VICTUS : {response}")

def main():
    chat = VictusChat()
    chat.start_chat()

if __name__ == "__main__":
    main() 