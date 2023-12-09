import React, { useState } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [selectedBooks, setSelectedBooks] = useState([]);
  const [allBooksSelected, setAllBooksSelected] = useState(false);

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (newMessage.trim() === '') return;

    const userMessage = { content: newMessage, sender: 'user' };
    const botMessage = { content: newMessage, sender: 'bot' };

    const updatedMessages = [...messages, userMessage, botMessage];
    setMessages(updatedMessages);
    setNewMessage('');
  };

  const handleBookCheckboxChange = (book) => {
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
      const allBooks = Array.from({ length: 10 }, (_, index) => `Book ${index + 1}`);
      setSelectedBooks(allBooks);
      setAllBooksSelected(true);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAA Chat Bot</h1>
      </header>
      <div className="App-content">
        <div className="ChatContainer">
          <div className="MessageList">
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
              <button style={{ width: '60px' }} type='submit' onClick={handleSendMessage}>Send</button>
            </form>
          </div>
        </div>
        <div className="BookSelectionContainer">
          <h2>Topic Selection</h2>
          <div>
            {Array.from({ length: 10 }, (_, index) => `Book ${index + 1}`).map((book, index) => (
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
          <div className="CheckboxContainer">
            <input type="checkbox" id="selectAll" onChange={handleSelectAllBooks} />
            <label htmlFor="selectAll">Select All</label>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
