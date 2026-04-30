import { useEffect, useRef, useState } from "react";
import "./App.css";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";

function App() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [input, setInput] = useState("");
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const BACKEND_URL = "http://localhost:8000";

  const inputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const activeChat = chats.find((chat) => chat.id === activeChatId);

  useEffect(() => {
    const storedChats = localStorage.getItem("engineeringChats");

    if (storedChats) {
      const parsedChats = JSON.parse(storedChats);
      setChats(parsedChats);

      if (parsedChats.length > 0 && !activeChatId) {
        setActiveChatId(parsedChats[0].id);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("engineeringChats", JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [activeChat?.messages, isBotTyping]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [activeChatId]);

  const createNewChat = () => {
    const newChat = {
      id: crypto.randomUUID(),
      title: `Chat ${chats.length + 1}`,
      messages: [],
      createdAt: new Date().toISOString(),
    };

    setChats((previousChats) => [newChat, ...previousChats]);
    setActiveChatId(newChat.id);
    setSearchTerm("");
    return newChat.id;
  };

  const selectChat = (chatId) => {
    setActiveChatId(chatId);
    setInput("");
  };

  const visibleChats = chats.filter((chat) =>
    chat.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isBotTyping) return;

    const userMessage = {
      id: crypto.randomUUID(),
      text: trimmed,
      sender: "user",
      timestamp: new Date().toISOString(),
    };

    let targetChatId = activeChatId;
    if (!activeChat) {
      targetChatId = crypto.randomUUID();
      const newChat = {
        id: targetChatId,
        title: `Chat ${chats.length + 1}`,
        messages: [userMessage],
        createdAt: new Date().toISOString(),
      };
      setChats((previousChats) => [newChat, ...previousChats]);
      setActiveChatId(targetChatId);
    } else {
      setChats((previousChats) =>
        previousChats.map((chat) =>
          chat.id === targetChatId
            ? { ...chat, messages: [...chat.messages, userMessage] }
            : chat
        )
      );
    }

    setInput("");
    setIsBotTyping(true);

    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: trimmed }),
      });

      if (!response.ok) throw new Error("Backend not responding");

      const data = await response.json();
      const botMessage = {
        id: crypto.randomUUID(),
        text: data.reply,
        sender: "bot",
        timestamp: new Date().toISOString(),
      };

      setChats((previousChats) =>
        previousChats.map((chat) =>
          chat.id === targetChatId
            ? { ...chat, messages: [...chat.messages, botMessage] }
            : chat
        )
      );
    } catch (error) {
      console.error("Error connecting to backend:", error);
    } finally {
      setIsBotTyping(false);
    }
  };

  return (
    <div className="app">
      <div className="appLayout">
        <Sidebar
          chats={visibleChats}
          activeChatId={activeChatId}
          createNewChat={createNewChat}
          selectChat={selectChat}
          searchTerm={searchTerm}
          setSearchTerm={setSearchTerm}
        />

        <ChatWindow
          messages={activeChat?.messages ?? []}
          input={input}
          setInput={setInput}
          isBotTyping={isBotTyping}
          handleSend={handleSend}
          messagesEndRef={messagesEndRef}
          inputRef={inputRef}
        />
      </div>
    </div>
  );
}

export default App;
