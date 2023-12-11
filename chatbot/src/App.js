import React, { useState, useRef, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import VisualizationPage from './Visualization'; 
import NavigationButton from './NavigationButton';


import './App.css';

function App() {

  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [selectedBooks, setSelectedBooks] = useState([]);
  const [allBooksSelected, setAllBooksSelected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [isChitChatFlag, setIsChitChatFlag] = useState(false);
  let globalChitChatFlag = true;
  const messageListRef = useRef(null);


  const allBooks = [
    'The Adventures of Sherlock Holmes',
    'Alice\'s Adventures in Wonderland',
    'And It Was Good',
    'Into the Primitive',
    'Pigs is Pigs',
    'The Fall of the House of Usher',
    'The Gift of the Magi',
    'The Jungle Book',
    'The Red Room',
    'Warrior of Two Worlds',
  ];
  const nameToNovel = {
    'sherlock': 'The Adventures of Sherlock Holmes',
    'alice': 'Alice\'s Adventures in Wonderland',
    'good': 'And It Was Good',
    'primitive': 'Into the Primitive',
    'pigs_is_pigs': 'Pigs is Pigs',
    'usher': 'The Fall of the House of Usher',
    'magi': 'The Gift of the Magi',
    'jungle': 'The Jungle Book',
    'redroom': 'The Red Room',
    'warrior': 'Warrior of Two Worlds'
  }
  const addMessage = async () => {
    const userMessage = { content: newMessage, sender: 'user' };
    const updatedMessages = [...messages, userMessage];
    console.log('updated messages : ', updatedMessages);
    setMessages(updatedMessages);
    setNewMessage('');
    console.log('in add message : ', messages)
  }

  const getNovelResponse = async () => {
    console.log(' in  getNovelResponse')
    let responseText;
    try {
      const response = await fetch('http://127.0.0.1:5000/novel', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_input: newMessage
        }),
      }).then(response => response.json())
        .then(data => {
          console.log('data[\'novel\'] : ', data['novel']);
          console.log('data[\'name\'] : ', data['name']);
          responseText = data['novel'];
          console.log('Novel name : ', nameToNovel[data['name']])
          console.log('selected : ', selectedBooks)
          const updatedSelectedBooks = [nameToNovel[data['name']]];
          setSelectedBooks(updatedSelectedBooks);
          const userMessage = { content: newMessage, sender: 'user' };
          const botMessage = { content: responseText, sender: 'bot' };
          const updatedMessages = [...messages, userMessage, botMessage];
          setMessages(updatedMessages);
          console.log('response added : ', messages)
        });;
    } catch (error) {
      console.error('Error:', error.message);
    } finally {
      setLoading(false);
    }
  }

  const getChitChatResponse = async () => {
    let responseText;
    try {
      const response = await fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_input: newMessage
        }),
      }).then(response => response.json())
        .then(data => {
          console.log('response', data['generated_rerereresponse']);
          responseText = data['generated_rerereresponse'];
          const userMessage = { content: newMessage, sender: 'user' };
          const botMessage = { content: responseText, sender: 'bot' };
          const updatedMessages = [...messages, userMessage, botMessage];
          setMessages(updatedMessages);
          console.log('response added : ', messages)
        });;
    } catch (error) {
      console.error('Error:', error.message);
    } finally {
      setLoading(false);
    }
  }

  const handleSendMessage = async (e) => {
    setLoading(true);
    e.preventDefault();
    if (newMessage.trim() === '') return;
    await addMessage()
    const flag = await isChitChat()
    console.log('globalChitChatFlag : ', globalChitChatFlag)
    if (globalChitChatFlag) {
      console.log('this is chit chat!')
      await getChitChatResponse()
    } else {
      console.log('this is a novel question!')
      await getNovelResponse()
    }
    globalChitChatFlag = true;
  };

  const isChitChat = async (e) => {
    try {
      console.log('in is chit chat')
      const response = await fetch('http://127.0.0.1:5000/chitchatclassifier', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_input: newMessage
        }),
      }).then(response => response.json())
        .then(data => {
          console.log('isChitChat response', data['isChitchat']);
          const chitChatFlag = data['isChitchat'];
          console.log('chitChatFlag : ', chitChatFlag)
          if (chitChatFlag === '1') {
            console.log('in here')
            globalChitChatFlag = true;
          } else {
            console.log('out here')
            globalChitChatFlag = false;
          }
          return chitChatFlag;
        });;
    } catch (error) {
      console.error('Error:', error.message);
    } finally {
    }
  };

  const handleBookCheckboxChange = (book) => {
    console.log(' in  handleBookCheckboxChange :', book)
    const updatedSelectedBooks = selectedBooks.includes(book)
      ? selectedBooks.filter((selectedBook) => selectedBook !== book)
      : [...selectedBooks, book];

    setSelectedBooks(updatedSelectedBooks);
  };

  const handleSelectAllBooks = () => {
    if (allBooksSelected) {
      setAllBooksSelected(false);
      setSelectedBooks([]);
    } else {
      setSelectedBooks(allBooks);
      setAllBooksSelected(true);
    }
  };

  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <Router>

      <div className="App">
        <header className="App-header">
          <h1>ChatGenix Bot</h1>
          <NavigationButton /> {}

        </header>
        <Routes>
          <Route path="/" element={
            <div className="App-content">
              <div className="ChatContainer">
                <div className="MessageList" ref={messageListRef}>
                  {messages.map((message, index) => (
                    <div
                      key={index}
                      className={`Message ${message.sender === 'user' ? 'UserMessage' : 'BotMessage'}`}
                    >
                      {message.content}
                    </div>
                  ))}
                </div>
                <div className="MessageInputContainer">
                  <form>
                    <input
                      type="text"
                      placeholder="Type a message..."
                      value={newMessage}
                      onChange={(e) => setNewMessage(e.target.value)}
                      style={{ width: '300px' }}
                    />
                    <button style={{ width: '60px' }} type='submit' onClick={handleSendMessage} disabled={loading} >{loading ? 'Loading' : 'Send'}</button>
                  </form>
                </div>
              </div>
              <div className="BookSelectionContainer">
                <h2>Topic Selection</h2>
                <div>
                  {allBooks.map((book, index) => (
                    <div key={index} className="CheckboxContainer">
                      <input
                        type="checkbox"
                        id={book}
                        checked={selectedBooks.includes(book)}
                        onChange={() => handleBookCheckboxChange(book)}
                      />
                      <label htmlFor={book}>{book}</label>
                    </div>
                  ))}
                </div>
                <br></br><br></br>
                <div className="CheckboxContainer">
                  <input type="checkbox" id="selectAll" onChange={handleSelectAllBooks} />
                  <label htmlFor="selectAll">Select All</label>
                </div>
              </div>
            </div>
          } />
          <Route path="/visualizations" element={<VisualizationPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
