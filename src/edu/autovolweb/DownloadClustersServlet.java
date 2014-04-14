package edu.autovolweb;

import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

/**
 * Servlet implementation class DownloadClustersServlet
 */
@WebServlet("/DownloadClustersServlet")
public class DownloadClustersServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String userId = request.getParameter("user");
		// Load cluster label mapping
		List<EMCluster> clusters = readClusterFile(userId, DataUploadServlet.CLUSTER_LABELS_FILE);
		
		Writer writer = response.getWriter();
		for (EMCluster c : clusters) {
			writer.write(c + "\n");
		}
		
		writer.write("\n\n AVGED \n\n");
		List<EMCluster> avgClusters = readClusterFile(userId, DataUploadServlet.AVG_CLUSTER_LABELS_FILE);
		for (EMCluster c : avgClusters) {
			writer.write(c + "\n");
		}
	}

	private List<EMCluster> readClusterFile(String userId, String clusterFile) throws IOException {
		byte[] encoded = Files.readAllBytes(Paths.get(
				DataUploadServlet.constructUserFileName(userId, clusterFile)));
		String json = Charset.defaultCharset().decode(
				ByteBuffer.wrap(encoded)).toString();
		
		Gson gson = new Gson();
		Type collectionType = new TypeToken<List<EMCluster>>(){}.getType();
		List<EMCluster> clusters = gson.fromJson(json, collectionType);
		return clusters;
	}

}
