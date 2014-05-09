package edu.autovolweb;

import java.io.IOException;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Servlet implementation class MainClassifyServlet
 */
@WebServlet("/MainClassifyServlet")
public class MainClassifyServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
	private static final String SERVLET = "/AvgRfClassifyServlet";

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		getServletContext().getRequestDispatcher(SERVLET).forward(request, response);
		
		//response.sendRedirect("AvgKnnClassifyServlet");
	}

}
