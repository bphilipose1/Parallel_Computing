
#pragma once  // only process the first time it is included; ignore otherwise
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <set>
#include <array>
#include <mpi.h>



template <int k, int d>
class KMeansMPI {
public:
    // some type definitions to make things easier
    typedef std::array<u_char,d> Element;
    class Cluster;
    typedef std::array<Cluster,k> Clusters;
    const int MAX_FIT_STEPS = 300;

    // debugging
    const bool VERBOSE = true;  // set to true for debugging output
#define V(stuff) if(VERBOSE) {using namespace std; stuff}

    /**
     * Expose the clusters to the client readonly.
     * @return clusters from latest call to fit()
     */
    virtual const Clusters& getClusters() {
        return clusters;
    }

    /**
     * fit() prepares for the k means algorithm with root process
    */
    virtual void fit(const Element *data, int data_n) {
        elements = data;
        n = data_n;
        V(cout << "Total DataSize: " << n << endl;)
        fitWork(0);
    }

    /**
     * fitwork is the main kMeans algorithm for all MPI processes
    */
    virtual void fitWork(int rank)  {
        if(rank == 0)   {
            reseedClusters(); 
            
        }
        bcastCentroids(rank);
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        dist.resize(n); 
        Clusters prior = clusters; 
        prior[0].centroid[0]++; 
        int generation = 0;
        distributeData();
        while (generation++ < MAX_FIT_STEPS && prior != clusters) {
            updateDistances();
            prior = clusters;
            updateClusters();
            mergeClusters(rank);
            bcastCentroids(rank);            
        }
    }
    

    /**
     * The algorithm constructs k clusters and attempts to populate them with like neighbors.
     * This inner class, Cluster, holds each cluster's centroid (mean) and the index of the objects
     * belonging to this cluster.
     */
    struct Cluster {
        Element centroid;  // the current center (mean) of the elements in the cluster
        std::vector<int> elements;

        // equality is just the centroids, regarless of elements
        friend bool operator==(const Cluster& left, const Cluster& right) {
            return left.centroid == right.centroid;  // equality means the same centroid, regardless of elements
        }
    };

protected:
    const Element *elements = nullptr;       // set of elements to classify into k categories (supplied to latest call to fit())
    int n = 0;                               // number of elements in this->elements
    Clusters clusters;                       // k clusters resulting from latest call to fit()
    std::vector<std::array<double,k>> dist;  // dist[i][j] is the distance from elements[i] to clusters[j].centroid 
    std::vector<std::vector<int>> localCentroidList;
    int startIndexSect;
    int endIndexSect;
    std::vector<Element> localElements;
    /**
     * Get the initial cluster centroids.
     * Default implementation here is to just pick k elements at random from the element
     * set
     * @return list of clusters made by using k random elements as the initial centroids
     */
    virtual void reseedClusters()   {
        for (int i = 0; i < k; i++) {
            clusters[i].centroid = elements[0]; // Assuming you want all centroids to start from the first element
            clusters[i].elements.clear();
        }
    }

    /**
     * Distributes elements data among MPI processes.
     * @return List of elements that MPI nodes can update clusters with.
     */
    void distributeData()   {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Give each MPI Node their own section
        std::vector<int> sendCounts(size, 0);
        std::vector<int> startIndex(size, 0);
        int elementsPerProcess = n / size;
        int remainder = n % size;

        // Prep Array of Counts Per Node (in bytes)
        for (int i = 0; i < size; i++) { 
            sendCounts[i] = elementsPerProcess;
            if (i == size - 1) { 
                sendCounts[i] += remainder;
            }
            
            if(rank == 0)   {V(cout << "Rank: " <<  rank << " SectSize: " << sendCounts[i] << endl;)}
            
            sendCounts[i] *= sizeof(Element); 
        }
        
        // Prep Array of Starting Indexes Per Node (in bytes)
        startIndex[0] = 0;
        for (int i = 1; i < size; i++) {
            startIndex[i] = startIndex[i - 1] + sendCounts[i - 1];
        }
        
        // Scatterv with byte passing
        localElements.resize(sendCounts[rank] / sizeof(Element));
        MPI_Scatterv(&elements[0], sendCounts.data(), startIndex.data(), MPI_BYTE, &localElements[0], sendCounts[rank], MPI_BYTE, 0, MPI_COMM_WORLD);
        startIndexSect = startIndex[rank]/sizeof(Element);
        endIndexSect = (startIndex[rank]+sendCounts[rank])/sizeof(Element);
        
        
    }

    /**
     * Updates the distance matrix based on current centroids with given section of elements.
     * @return Filled in distance matrix of centroids and localElements distance from them.
     */
    void updateDistances()  {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        //  Each node computes distances for its portion of elements
        for (int i = 0; i < (int)localElements.size(); i++) {
            for (int j = 0; j < k; j++) {
                dist[i][j] = distance(clusters[j].centroid, localElements[i]);
            }
        }
    }
    
    /**
     * Updates clusters with new elements based on the distance matrix.
     * @return Updated Centroids based on elements in its cluster.
     */
    void updateClusters()   {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Reinitialize Clusters, dont need last generations clusters
        for (int j = 0; j < k; j++) {
            clusters[j].centroid = Element{};
            clusters[j].elements.clear();
        }
        // For each element, put it in its closest cluster
        for (int i = 0; i < (int)localElements.size(); i++) {
            int min = 0;
            for (int j = 1; j < k; j++) {
                if (dist[i][j] < dist[i][min])  {
                    min = j;
                }
            }
            accum(clusters[min].centroid, clusters[min].elements.size(), localElements[i], 1);
            clusters[min].elements.push_back(i + startIndexSect);
            V(cout << "Rank" << rank << " localElements[" << i << "/" << (((int)localElements.size())-1) << "]: " << (i + startIndexSect) << endl;)
        }
    }

    /**
     * Obtains clusters from each MPI process and performs weighted average of centroid RGB values and stores in root node.
     * @param rank current MPI's processes rank.
     * @return List of elements that MPI nodes can update clusters with.
     */
    void mergeClusters(int rank)    { // DONE       
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Make space for k centroids, each with d dimensions and an element count
        std::vector<int> localData((k * (d + 1)), 0); 

        for(int centroidIndex = 0; centroidIndex < k; centroidIndex++) {
            int baseIndex = centroidIndex * (d + 1);
            localData[baseIndex] = clusters[centroidIndex].elements.size();
            for(int dimIndex = 0; dimIndex < d; dimIndex++) {
                localData[baseIndex + 1 + dimIndex] = clusters[centroidIndex].centroid[dimIndex];
            }
        }

        // Gather data at root
        std::vector<int> allData;
        if(rank == 0) {
            allData.resize((size * k * (d + 1)), 0);
        }

        MPI_Gather(localData.data(), k * (d + 1), MPI_INT, allData.data(), k * (d + 1), MPI_INT, 0, MPI_COMM_WORLD);

        // Root computes the weighted average
        if(rank == 0) {
            for(int i = 0; i < k; i++) {
                int totalWeight = 0;
                std::vector<int> weightedSum(d, 0);
                for(int x = 0; x < size; x++) {
                    int count = allData[x * k * (1 + d) + i * (1 + d)]; // How many elements contributed to centroid average (weight)
                    if(count > 0) { // Centroids that have no elements update it, have no say in centroid average
                        totalWeight += count;
                        for(int j = 0; j < d; j++) {
                            weightedSum[j] += (int)(allData[x * k * (1 + d) + i * (1 + d) + 1 + j]) * count; // Weighted average calculations
                        }
                    }
                }
                if(totalWeight > 0) { // Avoid division by zero (Rare Edge case of all centroids averaging out 0,0,0)
                    for(int j = 0; j < d; j++) {
                        clusters[i].centroid[j] = (int)(weightedSum[j] / (double)totalWeight); // Update global centroid
                    }
                }
            }
        }
        
        // Gather the elements for each centroid in Root Node
        gatherElements();
    }
    
    /**
     * Combines clusters elements from different MPI processes at root.
     * @return Root node contains the overall cluster of elements.
     */
    void gatherElements() {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Prepare data for sending
        std::vector<int> localElementsArr;
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < (int)clusters[i].elements.size(); j++) {
                localElementsArr.push_back(clusters[i].elements[j]);
                
            }
            localElementsArr.push_back(-1); // Delimiter between clusters
        }
        V(cout << "Rank #" << rank << " Packed Arr Szie: " << localElementsArr.size() << endl;)
        for(int x = 0; x < (int)localElementsArr.size(); x++)    {
            V(cout << "Rank:" << rank << "Value[" << x << "]: " << localElementsArr[x] << endl;)
        }

    
        int localSendCount = localElementsArr.size();

        // Gather the send counts at the root
        std::vector<int> recvCounts(size, 0);
        MPI_Gather(&localSendCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate total receive size and displacements for gathering at the root.
        std::vector<int> recvOffsets(size, 0);
        int totalRecvSize = 0;
        if(rank == 0) {
            for(int i = 0; i < size; i++) {
                recvOffsets[i] = totalRecvSize;
                totalRecvSize += recvCounts[i];
                V(cout << "Rank " << i << ": Offset: " << recvOffsets[i] << " Count: " << recvCounts[i] << endl;)
            }
        }
        std::vector<int> recvbuf(totalRecvSize, 0);

        MPI_Gatherv(localElementsArr.data(), localSendCount, MPI_INT,
                    recvbuf.data(), recvCounts.data(), recvOffsets.data(), MPI_INT, 0, MPI_COMM_WORLD);


        // At the root, process the received data.
        if(rank == 0) {
            for (int j = 0; j < k; j++) {
                clusters[j].elements.clear();
            }
            int currentCluster = 0;
            V(cout << "Expected totalRecv" << totalRecvSize << " Actual Size" << recvbuf.size() << endl;)
            for(int i = 0; i < totalRecvSize; i++) {
                V(cout << recvbuf[i] << " ";)
                if(recvbuf[i] == -1) {
                    currentCluster++;
                    if(currentCluster >= k)
                        currentCluster = 0;
                }
                else    {
                    clusters[currentCluster].elements.push_back(recvbuf[i]);
                }                
            }
        }
    }
    
        /**
     * Sends updated weighted clusters to all MPI processes.
     * @param rank current MPI's processes rank.
     * @return All MPI Processes recieve update overall Centroids.
     */
    void bcastCentroids(int rank) {
        int count = k * d;
        std::vector<u_char> buffer(count, 0);
        if(rank == 0) {
            int i = 0;
            for(int j = 0; j < k; j++) {
                for(int jd = 0; jd < d; jd++) {
                    buffer[i++] = clusters[j].centroid[jd];
                }
            }
        }

        //Use MPI_BYTE for broadcasting u_char data (1 Byte == 1 u_char type)
        MPI_Bcast(buffer.data(), count, MPI_BYTE, 0, MPI_COMM_WORLD);
        if(rank != 0) {
            //The non-root nodes update their centroids from the received buffer
            int i = 0;
            for(int j = 0; j < k; j++) {
                for(int jd = 0; jd < d; jd++) {
                    clusters[j].centroid[jd] = buffer[i++];
                }
            }
        }
    }

    /**
     * Method to update a centroid with an additional element(s)
     * @param centroid   accumulating mean of the elements in a cluster so far
     * @param centroid_n number of elements in the cluster so far
     * @param addend     another element(s) to be added; if multiple, addend is their mean
     * @param addend_n   number of addends represented in the addend argument
     */
    virtual void accum(Element& centroid, int centroid_n, const Element& addend, int addend_n) const {
        int new_n = centroid_n + addend_n;
        for (int i = 0; i < d; i++) {
            double new_total = (double) centroid[i] * centroid_n + (double) addend[i] * addend_n;
            centroid[i] = (u_char)(new_total / new_n);
        }
    }

    /**
     * Subclass-supplied method to calculate the distance between two elements
     * @param a one element
     * @param b another element
     * @return distance from a to b (or more abstract metric); distance(a,b) >= 0.0 always
     */
    virtual double distance(const Element& a, const Element& b) const = 0;

};

