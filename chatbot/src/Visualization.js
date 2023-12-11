import React, { useState, useEffect } from 'react';
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

import './Visualization.css'
function VisualizationPage() {
  const [images, setImages] = useState([]); // State to store image data

  useEffect(() => {
    // Fetch images data on component mount
    fetchImages();
  }, []);

  const fetchImages = async () => {
    // Fetch image data from your API or define it here
    try {
      await fetch('http://127.0.0.1:5000/visualization', {method: 'POST'});
    }catch{
      console.log("No fetch")
    }

    const imageData = [
      // Example data structure
      { src: '/bar_graph.png', explanation: 'We plotted a bar chart to show the 10 most common words which are  present in the user queries. Bar graphs can be used for large variations in data and can represent additional variables if needed, like the frequency of words over time\n' },
      { src: '/doughnut_chart.png', explanation: 'We plotted a doughnut graph to provide the count of how many queries are chit-chat queries and how many are novel queries fired by the user. Similar to pie charts, they emphasize the proportion of different types of queries—chit-chat versus novel-related' },
      { src: '/pie_chart.png', explanation: '\'We plotted a pie chart to show the distribution of user queries across different topics. This would help us understand which topics are more prevalent. They allow for immediate comparison of proportions, showing which topics are most and least common within the user queries.\n' },
      { src: 'response_time_chart.png', explanation: 'We also added a plot in the form of a line chart representing the response times for different queries, and  a horizontal line indicating the average response time.\n' +
            'Line charts excel at showing trends over time or across categories. Here, they can show how response times vary across different queries.\n' +
            'By including an average response time line, you can benchmark individual query response times against an average, identifying outliers and performance dips or spikes.If the data is time-based, line charts can show how response times have improved or worsened, providing insights into system performance or user experience trends\n' },

      // Add more images
    ];
    setImages(imageData);
  };

  const refreshImages = async () => {
    // Call the refresh API
    try {
      await fetch('http://refreshapi:3000/pppp', { method: 'POST' });
      fetchImages(); // Refetch images after refresh
    } catch (error) {
      console.error('Error refreshing images:', error);
    }
  };

  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1, // Show one image at a time
    slidesToScroll: 1,
    adaptiveHeight: false // Adjust the height of the carousel to the current slide
  };

  return (
      <div>
        <h2 style={{backgroundColor: "white"}}>Graphs for Visualization</h2>
        {/*<button onClick={refreshImages} style={{ width: '30%' }}>Refresh</button>*/}
        <Slider {...settings}>
          {images.map((img, index) => (
              <div key={index} className="image-card">
                <img src={img.src} alt={`Visualization ${index}`} />
                <div className="explanation">{img.explanation}</div>
              </div>
          ))}
        </Slider>
      </div>
  );
}

export default VisualizationPage;
