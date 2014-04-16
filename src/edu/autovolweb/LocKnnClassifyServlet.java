package edu.autovolweb;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;

import weka.clusterers.FilteredClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

/**
 * Servlet implementation class LocKnnClassifyServlet
 */
@WebServlet("/LocKnnClassifyServlet")
public class LocKnnClassifyServlet extends BaseKnnClassify {
	private static final long serialVersionUID = 1L;
	
	
	private Map<String, FilteredClusterer> locClustererMap;
	private Map<String, List<String>> locClustersMap;

	// TODO: save this data into files after we do orig cluster?
	@Override
	protected Instances loadData(String userId) throws Exception {
		Instances allData = ViewDataServlet.loadAllData(userId, this);

		Instances allDataLoc = CurrentStateUtil.replaceLocationData(allData, 
				new int[] {2,3,4}, getLocClusters(userId), 
				((SimpleKMeans) getLocClusterer(userId).getClusterer()).getAssignments());
		allDataLoc.setClass(allDataLoc.attribute("ringer"));

		return allDataLoc;

	}
	
	@Override
	public void init() throws ServletException {
		super.init();
		
		locClustererMap = new ConcurrentHashMap<>();
		locClustersMap = new ConcurrentHashMap<>();
	}
	
	@Override
	protected void initForUser(String userId) throws Exception {
		FilteredClusterer locClusterer = (FilteredClusterer) SerializationHelper.read(new FileInputStream(
				DataUploadServlet.constructUserFileName(userId,DataUploadServlet.LOC_CLUSTERER_FILE)));
		locClustererMap.put(userId, locClusterer);

		Gson gson = new Gson();
		String locJson = getFileJson(userId, DataUploadServlet.LOC_DATA_FILE);
		Type locCollectionType = new TypeToken<List<String>>(){}.getType();
		List<String> locClusters = gson.fromJson(locJson, locCollectionType);
		locClustersMap.put(userId, locClusters);
		super.initForUser(userId);
	}

	@Override
	protected Instance constructTarget(String input, String userId) {
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);

		Instance locTarget = CurrentStateUtil.extractLocInstance(state);
		String locCluster;
		try {
			locCluster = "" + ((int) getLocClusterer(userId).clusterInstance(locTarget));
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		if (!getLocClusters(userId).contains(locCluster)) {
			locCluster = "other";
		}
		
		Instance target = CurrentStateUtil.toLocInstance(state,
				locCluster, getLocClusters(userId));
		
		return target;
	}
	
	@Override
	protected String prepareOutput(JsonObject json, Instance target) {
		json.addProperty("loc", target.value(target.dataset().attribute("loc")));
		return json.toString();
	}
	
	protected FilteredClusterer getLocClusterer(String userId) {
		return locClustererMap.get(userId);
	}
	
	protected List<String> getLocClusters(String userId) {
		return locClustersMap.get(userId);
	}
	
	private String getFileJson(String userId, String filename) throws IOException {
		String fullFilename = DataUploadServlet.constructUserFileName(userId, filename);
		byte[] encoded = Files.readAllBytes(Paths.get(fullFilename));
		String json = Charset.defaultCharset().decode(
				ByteBuffer.wrap(encoded)).toString();
		return json;
	}
}
