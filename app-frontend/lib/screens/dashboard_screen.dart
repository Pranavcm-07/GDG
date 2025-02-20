import 'package:flutter/material.dart';

class DashboardScreen extends StatefulWidget {
  final Map<String, dynamic> data; // Accept JSON data

  const DashboardScreen({Key? key, required this.data}) : super(key: key);
  @override
  _DashboardScreenState createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
    appBar: AppBar(
      backgroundColor: Colors.white,
      elevation: 0,
      leading: IconButton(
        icon: Icon(Icons.arrow_back, color: Color.fromARGB(255, 51, 85, 255), size: 30),
        onPressed: () {
          Navigator.pop(context);
        },
      ),
    ),
      body: SingleChildScrollView(
  child: Container(
    color: Colors.white,
    padding: const EdgeInsets.symmetric(horizontal: 20.0),
    child: Column(
      mainAxisAlignment: MainAxisAlignment.start,
      children: [
        Row(
              mainAxisAlignment: MainAxisAlignment.start, // Aligns the text to the left
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                  "Dashboard",
                  style: TextStyle(
                    color: Color.fromARGB(255, 51, 85, 255),
                    fontWeight: FontWeight.w900,
                    fontFamily: "Roboto",
                    fontSize: 30,
                  ),
                ),
                Text(
                  "Your data, your decisions",
                  style: TextStyle(
                    color: Colors.black,
                    fontFamily: "Roboto",
                    fontSize: 14,
                    fontWeight: FontWeight.w600
                  ),
                )
                  ],
                )
              ],
            ),
        // User Profile Section
        SizedBox(height: 20),
        Container(
          padding: EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Color.fromARGB(255, 51, 85, 255),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Row(
            children: [
              CircleAvatar(
                radius: 40,
                backgroundColor: Colors.white,
                child: Icon(Icons.person, size: 40, color: Color.fromARGB(255, 51, 85, 255)),
              ),
              SizedBox(width: 20),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('User Name', style: TextStyle(color: Colors.white, fontSize: 18)),
                  SizedBox(height: 5),
                  Text('user@example.com', style: TextStyle(color: Colors.white70)),
                ],
              ),
            ],
          ),
        ),
        SizedBox(height: 20),
        // Stats Section (Example: Sales, Revenue, etc.)
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Expanded(child: StatCard(title: 'Chapter', value: widget.data["Best Matched GDPR Rule"][0].toString())),
            SizedBox(width: 10),
            Expanded(child: StatCard(title: 'Article No.', value: widget.data["Best Matched GDPR Rule"][1].toString())),
          ],
        ),
        SizedBox(height: 10),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(child: StatCard(title: 'Sub Article', value: widget.data["Best Matched GDPR Rule"][2].toString())),
            SizedBox(width: 10),
            Expanded(child: StatCard(title: 'Score', value: (widget.data["Similarity Score"] as num?)?.toStringAsFixed(3) ?? "0.000")),
          ],
        ),

        SizedBox(height: 20),
        // Placeholder for additional content (e.g. charts, reports)
      Container(
        padding: EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(10),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.3),
              spreadRadius: 2,
              blurRadius: 5,
              offset: Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "GDPR Summary",
              style: TextStyle(
                color: Color.fromARGB(255, 51, 85, 255),
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 8),
            Text(
              widget.data["GDPR Summary"].replaceAll('*', ''),
              style: TextStyle(fontSize: 14, color: Colors.black87),
              softWrap: true,
            ),
          ],
        ),
      ),

      SizedBox(height: 20),
      // Gemini Analysis Section
      Container(
        padding: EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(10),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.3),
              spreadRadius: 2,
              blurRadius: 5,
              offset: Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "Gemini Analysis",
              style: TextStyle(
                color: Color.fromARGB(255, 51, 85, 255),
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 8),
            Text(
              widget.data["Gemini Analysis"].replaceAll('*', ''),
              style: TextStyle(fontSize: 14, color: Colors.black87),
              softWrap: true,
            ),
          ],
        ),
      ),
        SizedBox(height: 20),
      ],
    ),
  ),
)
    );
  }
}

// StatCard Widget for Dashboard Statistics
class StatCard extends StatelessWidget {
  final String title;
  final String value;

  const StatCard({required this.title, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.3), // Shadow color
            spreadRadius: 2, // Spread radius
            blurRadius: 5, // Blur radius
            offset: Offset(0, 4), // Shadow position (horizontal, vertical)
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(title, style: TextStyle(color: Color.fromARGB(255, 51, 85, 255), fontSize: 18)),
          SizedBox(height: 5),
          Text(value, style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}