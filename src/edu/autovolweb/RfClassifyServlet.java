package edu.autovolweb;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

/**
 * Servlet implementation class RfClassifyServlet
 */
@WebServlet("/RfClassifyServlet")
public class RfClassifyServlet extends ClearingHttpServlet {
	private static final long serialVersionUID = 1L;
	
	private Map<String, RandomForest> rfMap;
	private Set<String> initializedUsers;
	
	@Override
	public void init() throws ServletException {
		rfMap = new HashMap<>();
		initializedUsers = Collections.synchronizedSet(new HashSet<String>());
	}


	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String userId = request.getParameter("user");
		if (userId == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
			return;
		}
		if (!initializedUsers.contains(userId)) {
			try {
				initForUser(userId);
			} catch (Exception e) {
				e.printStackTrace();
				response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
				return;
			}
			
		}
		initializedUsers.add(userId);
		String input = request.getParameter("target");
		if (input == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
			return;
		}
		
		Instance target = constructTarget(input, userId);
		if (target == null) {
			response.sendError(HttpServletResponse.SC_BAD_REQUEST);
			return;
		}
		JsonObject json = new JsonObject();
		
		RandomForest rf = rfMap.get(userId);
		double classification;
		try {
			classification = rf.classifyInstance(target);
			String classStr = target.classAttribute().value((int) classification);
			json.addProperty("rf",classStr);
		} catch (Exception e) {
			e.printStackTrace();
			json.addProperty("rf","err");
		}
		response.getWriter().write(json.toString());
	}
	
	protected Instance constructTarget(String input, String userId) {
		Gson gson = new Gson();
		CurrentStateData state = gson.fromJson(input, CurrentStateData.class);
		return CurrentStateUtil.toInstance(state);
	}
	
	protected void initForUser(String userId) throws Exception {
		Instances data = loadData(userId);
		RandomForest rf = new RandomForest();
		rfMap.put(userId, rf);
		
		rf.buildClassifier(data);
	}

	protected Instances loadData(String userId) throws Exception {
		Instances data = ViewDataServlet.loadAllData(userId, this);
		data.setClass(data.attribute("ringer"));
		return data;
	}


	@Override
	protected void clear() {
		rfMap.clear();
		initializedUsers.clear();	
	}


}
