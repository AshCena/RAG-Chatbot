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
      { src: '/bar_graph.png', explanation: 'Explanation for Image 1' },
      { src: '/doughnut_chart.png', explanation: 'Explanation for Image 2' },
      { src: '/pie_chart.png', explanation: 'Explanation for Image 2' },
      { src: 'response_time_chart.png', explanation: 'Explanation for Image 2' },

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
        <h2>Graphs for Visualization</h2>
        <button onClick={refreshImages} style={{ width: '30%' }}>Refresh</button>
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
