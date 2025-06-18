let mapCenter = [27.6876589, 85.2897551];

let map = L.map('map', {
    center: mapCenter,
    zoom: 15, // Initial zoom level
    maxZoom: 20, // Increase max zoom level
    minZoom: 10, // Set a minimum zoom level (optional)
    scrollWheelZoom: true // Enable smooth zooming
});

// Add OpenStreetMap tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

// Define TIFF overlays and their bounds
let tiffLayers = [
    {
        url: 'http://127.0.0.1:8000/static/downsampled.png', // First TIFF PNG
        bounds: [[27.6336405, 85.3330463], [27.6481854, 85.3454091]] // First TIFF bounds
    },
    {
     url: 'http://127.0.0.1:8000/static/advancedown.png', // Second TIFF PNG
        bounds: [[27.6838582,  85.2858866], [27.6914596, 85.2936236]] // Second TIFF bounds
    }
];

// Add TIFF images as overlays
tiffLayers.forEach(tiff => {
    L.imageOverlay(tiff.url, tiff.bounds).addTo(map);
});

// Add drawing functionality
let drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

let drawControl = new L.Control.Draw({
    draw: {
        rectangle: true,
        polygon: false,
        circle: false,
        marker: false,
        polyline: false,
    },
    edit: {
        featureGroup: drawnItems,
        remove: true,
    },
});
map.addControl(drawControl);

let selectedBounds;
map.on(L.Draw.Event.CREATED, function (e) {
    drawnItems.clearLayers();
    selectedBounds = e.layer.getBounds();
    drawnItems.addLayer(e.layer);
});

// Button to recenter the map
document.getElementById('recenter-map').addEventListener('click', () => {
    map.setView(mapCenter, 15);
});

// Button to clear selection
document.getElementById('clear-selection').addEventListener('click', () => {
    drawnItems.clearLayers();
    selectedBounds = null;
});

// Button to start prediction
document.getElementById('go-to-prediction').addEventListener('click', async () => {
    if (!selectedBounds) {
        alert('Please select an area first.');
        return;
    }

    // Show loading overlay
    document.getElementById('loading-overlay').style.display = 'flex';

    // Hide the cropping window and show the prediction window
    document.getElementById('cropping-window').classList.remove('active');
    document.getElementById('prediction-window').classList.add('active');

    await handlePrediction();
});

// Function to handle prediction request
async function handlePrediction() {
    try {
        let response = await fetch('http://127.0.0.1:8000/crop_and_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topLeft: [selectedBounds.getNorthWest().lat, selectedBounds.getNorthWest().lng],
                bottomRight: [selectedBounds.getSouthEast().lat, selectedBounds.getSouthEast().lng]
            })
        });

        let data = await response.json();

        if (response.status === 400) {
            alert(data.detail);
            document.getElementById('loading-overlay').style.display = 'none';
            return;
        }

        // Display prediction results
        document.getElementById('cropped-image').src = `data:image/jpeg;base64,${data.original_image}`;
        document.getElementById('segmentation-mask').src = `data:image/png;base64,${data.segmentation_map}`;
        document.getElementById('blended-image').src = `data:image/png;base64,${data.blended_image}`;

        // Populate the table
        let tableHTML = "<tr><th>Category</th><th>Pixels</th><th>Area (ha)</th><th>Proportion (%)</th></tr>";
        for (let [category, stats] of Object.entries(data.area_statistics)) {
            tableHTML += `<tr>
                <td>${category}</td>
                <td>${stats.pixels}</td>
                <td>${stats.area_km2.toFixed(4)}</td>
                <td>${(stats.proportion * 100).toFixed(2)}</td>
            </tr>`;
        }
        document.getElementById('area-table').innerHTML = tableHTML;

        // Generate pie chart
        let chartColors = ['rgb(255, 162, 0)', 'rgb(255, 13, 0)', 'rgb(160, 32, 240)', 'rgb(18, 107, 0)', 'rgb(0, 110, 255)'];
        new Chart(document.getElementById('pie-chart'), {
            type: 'pie',
            data: {
                labels: Object.keys(data.area_statistics),
                datasets: [{
                    data: Object.values(data.area_statistics).map(stat => stat.proportion * 100),
                    backgroundColor: chartColors,
                    borderColor: '#fff',
                    borderWidth: 2,
                }],
            },
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: 'Land Cover Proportion (%)'
                    }
                }
            }
        });
    } catch (error) {
        alert('Error processing segmentation. Please try again.');
    } finally {
        // Hide the loading overlay
        document.getElementById('loading-overlay').style.display = 'none';
    }
}

// Function to reset results
function resetResults() {
    selectedBounds = null;
    drawnItems.clearLayers();
    document.getElementById('cropped-image').src = '';
    document.getElementById('segmentation-mask').src = '';
    document.getElementById('blended-image').src = '';
    document.getElementById('area-table').innerHTML = '';
    
    // Destroy previous pie chart instances
    Chart.helpers.each(Chart.instances, function (instance) {
        instance.destroy();
    });
}

// Button to download results
document.getElementById('download-images').addEventListener('click', () => {
    window.location.href = 'http://127.0.0.1:8000/download_zip';
});

// Button to go back to cropping
document.getElementById('back-to-cropping').addEventListener('click', () => {
    resetResults();
    document.getElementById('prediction-window').classList.remove('active');
    document.getElementById('cropping-window').classList.add('active');
});
