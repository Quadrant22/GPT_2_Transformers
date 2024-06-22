import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a pretrained model and tokenizer (e.g., GPT-2)
model_name = "gpt2"  # You can replace this with your desired model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a function to generate responses
def get_response():
    user_input = user_entry.get()
    if user_input:
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        with torch.no_grad():
            bot_output = model.generate(input_ids, max_length=100, num_return_sequences=1)
        bot_response = tokenizer.decode(bot_output[0], skip_special_tokens=True)
        conversation_text.config(state=tk.NORMAL)
        conversation_text.insert(tk.END, f"You: {user_input}\nBot: {bot_response}\n")
        conversation_text.config(state=tk.DISABLED)
        user_entry.delete(0, tk.END)

# Create the main GUI window
window = tk.Tk()
window.title("Chatbot")

# Create a conversation history text widget
conversation_text = tk.Text(window, wrap=tk.WORD, state=tk.DISABLED)
conversation_text.pack()

# Create a user input entry field
user_entry = tk.Entry(window, width=40)
user_entry.pack()

# Create a button to send user input
send_button = tk.Button(window, text="Send", command=get_response)
send_button.pack()

# Start the Tkinter main loop
window.mainloop()

