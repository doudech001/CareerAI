function Sidebar({ chats, activeChatId, createNewChat, selectChat, searchTerm, setSearchTerm }) {
  return (
    <aside className="sidebar">
      <div className="sidebarInner">
        <div className="branding">
          <div className="brandIcon">
            <span className="material-symbols-outlined">memory</span>
          </div>
          <div>
            <h1>AI ASSISTANT</h1>
          </div>
        </div>

        <button className="newChatButton" onClick={createNewChat}>
          <span className="material-symbols-outlined">add</span>
          New Chat
        </button>

        <div className="searchWrapper">
          <span className="material-symbols-outlined searchIcon">search</span>
          <input
            type="text"
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            placeholder="Search threads..."
          />
        </div>

        <nav className="chatList">
          <div className="sectionLabel">Recent History</div>
          {chats.length > 0 ? (
            chats.map((chat) => (
              <button
                key={chat.id}
                className={`chatItem ${chat.id === activeChatId ? "active" : ""}`}
                onClick={() => selectChat(chat.id)}
              >
                <span className="material-symbols-outlined chatIcon">chat_bubble</span>
                <span className="chatTitle">{chat.title}</span>
              </button>
            ))
          ) : (
            <div className="emptyChatList">No chats found</div>
          )}
        </nav>

        <div className="footerNav">
          <button className="footerButton">
            <span className="material-symbols-outlined">settings</span>
            Settings
          </button>

          <div className="profileCard">
            <img
              alt="AI Avatar"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuBKp_rV7mMOzNn0pqmIHqGl6AfH897h1V_Rn6IS7ARJueEQ52mSe7CwMmZR4eNaDk8HJjgAxFyCorBUqeUFplJlm25AgbWNsTT2TtGdnqzqDw9don_MK_a_BvBI2R6BwOTjnj2id4I6inDmH_vkGOImhAJuLeGkcH1BNXfQloomxWsVid6Ft5ofGCHd3WmI6pGiCGLZUmArikQhY1DsKJu3ELqS40Ri7Lyt_rBpbaCxdEIygCng8lsVSYi76k1CiBl1vVyTINEkdPW1"
            />
            <div>
              <span>Lead Engineer</span>
              <span>pro_user_882</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
