import streamlit as st 

st.title("Departments")
tab_titles = ["HR Support", "IT Support", "Transportation"]
tabs = st.tabs(tab_titles)

with tabs[0]:
    st.header("HR Support Tickets")
    for ticket in st.session_state["HR_tickets"]:
        ticket_str = str(ticket)
        # st.write(ticket)
        st.write(str(st.session_state["HR_tickets"].index(ticket)+1)+ " : " + ticket_str)

with tabs[1]:
    st.header("IT Support Tickets")
    for ticket in st.session_state["IT_tickets"]:
        # st.write(ticket)
        ticket_str = str(ticket)
        st.write(str(st.session_state["IT_tickets"].index(ticket)+1)+ " : " + ticket_str)

with tabs[2]:
    st.header("Transportation Tickets")
    for ticket in st.session_state["Transport_tickets"]:
        # st.write(ticket)
        ticket_str = str(ticket)
        st.write(str(st.session_state["Transport_tickets"].index(ticket)+1)+ " : " + ticket_str)
