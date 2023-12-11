// NavigationButton.js
import React from 'react';
import { useNavigate } from 'react-router-dom';

function NavigationButton() {
    const navigate = useNavigate();

    const handleVisualizationClick = () => {
        navigate('/visualizations');
    };

    return <button  onClick={handleVisualizationClick}  style={{ width: '30%' }} >Visualizations</button>;

}

export default NavigationButton;
