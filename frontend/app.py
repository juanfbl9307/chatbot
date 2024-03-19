import streamlit as st
import requests
import uuid

backend_chatbot_url = "http://localhost:8082/chat"


# Function to initialize session ID
def init_session_id():
    session_id = uuid.uuid4()
    st.session_state['session_id'] = str(session_id)


# Initialize session ID if not already initialized
if 'session_id' not in st.session_state:
    init_session_id()


# Function for generating LLM response
def generate_response(prompt_input):
    response_chat = requests.request("POST", backend_chatbot_url, headers=llm_headers, json={"content": prompt_input})
    if response_chat.status_code == 200:
        resp_json = response_chat.json()
        st.sidebar.json(resp_json)
        send_resp = str(resp_json['Response']).strip()
        return send_resp
    else:
        print(response_chat.text)
        return "Algo fallo en la respuesta.."


# Check if 'messages' is already in st.session_state, if not initialize it
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Bienvenido a tu asistente de pedidos, ¿en qué puedo ayudarte hoy?"}]

st.set_page_config(page_title="Asistente de negocios")
st.sidebar.title("Creado por Juan Felipe para todos los negocios")
st.sidebar.info(st.session_state['session_id'])
st.sidebar.warning("Refrescar la pagina restablecerá la conversación, cambiando el session_id")

llm_headers = {
    'session_id': st.session_state['session_id']
}

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# User-provided prompt
if user_prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt, unsafe_allow_html=True)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                response = generate_response(user_prompt)
                st.markdown(response, unsafe_allow_html=True)
            # message_placeholder = st.empty()
        #     full_response = ""
        #     with st.spinner("Thinking..."):
        #         response = generate_response(user_prompt)
        #     for chunk in response.split():
        #         full_response += chunk + " "
        #         time.sleep(0.05)
        #         message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
        #     message_placeholder.markdown(full_response, unsafe_allow_html=True)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
