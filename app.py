# import for typing
from langchain.chains import RetrievalQAWithSourcesChain

# gradio
import gradio as gr

global qa 
from qa import qa


#####
#
# Gradio fns
####

def create_gradio_interface(qa:RetrievalQAWithSourcesChain):
    """
    Create a gradio interface for the QA model
    """
    def add_text(history, text):
        history = history + [(text, None)]
        return history, ""

    def bot(history):
        response = infer(history[-1][0], history)
        sources = [doc.metadata.get("source") for doc in response['source_documents']]
        src_list = '\n'.join(sources)
        print_this = response['answer'] + "\n\n\n Sources: \n\n\n" + src_list


        history[-1][1] = print_this #response['answer']
        return history

    def infer(question, history):
        query =  question
        result = qa({"query": query, "history": history, "question": question})
        return result

    def vote(data: gr.LikeData):
        if data.liked:
            print("You upvoted this response: ")
        else:
            print("You downvoted this response: ")

    css="""
    #col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
    """

    title = """
    <div style="text-align: center;max-width: 1920px;">
        <h1>Chat with your Documentation</h1>
        <p style="text-align: center;">This is a privately hosted Docs AI Buddy ;)</p>
    </div>
    """

    head_style = """
    <style>
    @media (min-width: 1536px)
    {
        .gradio-container {
            min-width: var(--size-full) !important;
        }
    }
    </style>
    """

    with gr.Blocks(title="DocsBuddy AI 🤵🏻‍♂️", head=head_style) as demo:
        with gr.Column(min_width=900, elem_id="col-container"):
            gr.HTML(title)      
            with gr.Row():
                question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
            chatbot = gr.Chatbot([], elem_id="chatbot", label="DocuBuddy 🤵🏻‍♂️", height=660)
            #with gr.Row():
            #    clear = gr.Button("Clear")
            chatbot.like(vote, None, None)

            with gr.Row():
                clear = gr.ClearButton([chatbot, question])

        question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
            bot, chatbot, chatbot
        )
        #clear.click(lambda: None, None, chatbot, queue=False)
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface(qa)
    demo.queue().launch()
