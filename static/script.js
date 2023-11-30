    let nodes = [];
    let links = [];
    let startNodeLabel = '';
    let goalNodeLabel = '';

    const svg = d3.select("#graph-container")
        .append("svg")
        .attr("width", 900)
        .attr("height", 700);

    const g = svg.append("g");

    const r = 10; // Radius of nodes

    const zoom = d3.zoom()
        .extent([[0, 0], [1200, 800]])
        .scaleExtent([1, 8])
        .on("zoom", zoomed);

    svg.call(zoom);

    function zoomed({ transform }) {
        g.attr("transform", transform);
    }


    function drawGraph() {
        links.forEach(({ source, target, distance }) => {

            const centerX = (nodes[source].x + nodes[target].x) / 2;
            const centerY = (nodes[source].y + nodes[target].y) / 2;

            g.append("line")
                .attr("id", `line-${source}-${target}`) // Assign unique ID
                .attr("x1", nodes[source].x)
                .attr("y1", nodes[source].y)
                .attr("x2", nodes[target].x)
                .attr("y2", nodes[target].y)
                .attr("stroke", "black");

            g.append("text")
                .attr("x", centerX)
                .attr("y", centerY)
                .attr("dy", ".35em")
                .attr("text-anchor", "middle")
                .attr("class", "distance-label")
                .text(Math.round(distance));
        });

        nodes.forEach(({ x, y, label, index }) => {
            g.append("circle")
                .attr("id", `circle-${index}`)
                .attr("cx", x)
                .attr("cy", y)
                .attr("r", 20)
                .attr("fill", "white")
                .attr("stroke", "black");

            g.append("text")
                .attr("x", x)
                .attr("y", y)
                .attr("dy", ".35em")
                .attr("text-anchor", "middle")
                .attr("class", "node-label")
                .text(label);
        });

        console.log(nodes, links);
    }

    function generateGraph() {
        g.selectAll("*").remove();
        const nodeCount = +document.getElementById("node-count").value;

        fetch('/generate_nodes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `node_count=${nodeCount}`
        })
        .then(response => response.json())
        .then(data => {
            nodes = data.nodes;
            links = data.links;
            startNodeLabel = data.startNodeLabel;
            goalNodeLabel = data.goalNodeLabel;
            drawGraph();
        })
        .catch(error => console.error('Error:', error));
    }


// Create a slider input
let slider = document.getElementById("speedSlider");

function visualizePaths(visited_labels, paths, iterationCounts) {
    console.log(iterationCounts);
    console.log(paths);
    const selectedAlgo = document.getElementById("search_algo").value;
    let currentIndex = 0;

    let timer = setInterval(() => {
        const currentPath = paths[currentIndex];
        if (currentPath) {
            const sourceNode = nodes.find(node => node.label === currentPath[0]);
            const targetNode = nodes.find(node => node.label === currentPath[1]);
            if (sourceNode && targetNode) {
                // Check if the current node has been visited before (backtracking)
                // Backtracking is when the current node has been visited before
                const repeatedPathIndex = paths.slice(0, currentIndex).findIndex(path => path[0] === sourceNode.label);
                if (repeatedPathIndex !== -1 && selectedAlgo !== 'bfs') {
                    const lastPath = paths[repeatedPathIndex];
                    const lastSourceNode = nodes.find(node => node.label === lastPath[0]);
                    const lastTargetNode = nodes.find(node => node.label === lastPath[1]);
                    const lastLine = g.select(`#line-${lastSourceNode.index}-${lastTargetNode.index}`);
                    lastLine.attr("stroke", "black");
                    lastLine.attr("stroke-width", 1);
                }

                const line = g.select(`#line-${sourceNode.index}-${targetNode.index}`);
                const circle = g.select(`#circle-${targetNode.index}`);
                line.attr("stroke", "red");
                circle.attr("fill", "red");
                line.attr("stroke-width", 10);
                console.log(`From Node: ${sourceNode.label} to Node: ${targetNode.label}`);
            }
        }

        // Check if all paths are traversed
        if (currentIndex >= paths.length - 1) {
            console.log('Goal reached!');
            createTableHeader();
            clearInterval(timer);
            updateIterationTable(iterationCounts);
        }

        currentIndex++;
    }, document.getElementById("speedSlider").value);
}



    
    

// Add a new function to update the iteration table
function updateIterationTable(iterationCounts) {
    const table = document.getElementById('iterationTable');

    // Clear existing rows
    while (table.rows.length > 0) {
        table.deleteRow(0);
    }

    // Check if iterationCounts is defined
    if (iterationCounts) {
        // Add a header row
        const headerRow = table.insertRow();
        headerRow.innerHTML = "<th>Parameter</th><th>Value</th>";

        // Add rows for each parameter in iterationCounts
        for (const [param, value] of Object.entries(iterationCounts)) {
            const row = table.insertRow();
            const cellParam = row.insertCell(0);
            const cellValue = row.insertCell(1);

            // Remove underscores and capitalize the first letter of each word in the parameter name
            const formattedParam = param.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');

            // Round the value to a whole number if it's the path cost
            const formattedValue = (param === 'path_cost') ? Math.round(value) : value;

            cellParam.innerHTML = formattedParam;
            cellValue.innerHTML = formattedValue;
        }
    } else {
        console.log('iterationCounts is undefined');
    }
}



// Call this function to create a table header
function createTableHeader() {
    const table = document.getElementById('iterationTable');
    const headerRow = table.insertRow();
    const headers = ['Parameter', 'Value'];

    for (let header of headers) {
        const th = document.createElement('th');
        th.innerHTML = header;
        headerRow.appendChild(th);
    }
}


    
    
    






document.getElementById("run-search").addEventListener("click", function() {
    const startNodeLabel = document.getElementById("start-node").value;
    const goalNodeLabel = document.getElementById("goal-node").value;
    const selectedAlgorithm = document.getElementById("search_algo").value;

    console.log(`Sending ${selectedAlgorithm} request to server...`);

    fetch(`/${selectedAlgorithm}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `start_label=${startNodeLabel}&goal_label=${goalNodeLabel}`
    })
    .then(response => response.json())
    .then(data => {
        const visited_labels = data.visited_labels;
        const paths = data.paths;
        const iterationCounts = data.iteration_counts;
        visualizePaths(visited_labels, paths, iterationCounts);
    })
    .catch(error => console.error('Error:', error));
});











    function colorStartAndGoalNodes() {
        const startNodeLabel = document.getElementById("start-node").value;
        const goalNodeLabel = document.getElementById("goal-node").value;
        console.log("startNodeLabel:", startNodeLabel); // Add this line
        console.log("goalNodeLabel:", goalNodeLabel); // Add this line

        let startNodeFound = false;
        let goalNodeFound = false;

        // Clear existing red paths
        g.selectAll("line")
        .attr("stroke", "black")
        .attr("stroke-width", 1);

        nodes.forEach(node => {
            node.isStartNode = node.label === startNodeLabel;
            node.isGoalNode = node.label === goalNodeLabel;

            if (node.isStartNode) {
                startNodeFound = true;
            }

            if (node.isGoalNode) {
                goalNodeFound = true;
            }
        });

        console.log(startNodeFound); // Add this line
        console.log(goalNodeFound); // Add this line

        if (startNodeFound && goalNodeFound) {
            nodes.forEach(node => {
                const circle = document.getElementById(`circle-${node.index}`);
                if (circle) {
                    circle.setAttribute("fill", node.isStartNode ? "red" : node.isGoalNode ? "green" : "white");
                }
            });
        } else {
            alert("Start node or goal node not found in the generated graph.");
        }
    }



    document.getElementById("generate-graph").addEventListener("click", generateGraph);
    document.getElementById("run-search").addEventListener("click", colorStartAndGoalNodes);



    window.onload = function() {
        fetch('/graph_data')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data && data.nodes && data.links) {
                    nodes = data.nodes;
                    links = data.links;
                    startNodeLabel = data.startNodeLabel;
                    goalNodeLabel = data.goalNodeLabel;
                    drawGraph();
                } else {
                    console.error('Invalid data:', data);
                }
            })
            .catch(error => {
                console.error('Fetch error:', error);
            });
    }


    