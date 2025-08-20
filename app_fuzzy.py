import streamlit as st

# ---- Simple password gate ----
def check_password():
    def password_entered():
        if st.session_state["username"] in st.secrets["passwords"] and            st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # remove password from memory
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        st.error("😕 Incorrect username or password")
        return False
    else:
        return True

if not check_password():
    st.stop()

st.title("Fuzzy Quadrant Mapper")
st.write("This is a placeholder of the app. Full analysis logic should be here.")
