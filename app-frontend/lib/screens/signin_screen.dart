// import 'package:firebase_auth/firebase_auth.dart';
// // import 'package:firebase_signin/reusable_widgets/reusable_widget.dart';
// import 'package:flutter_application_1/reusable_widgets/reusable_widget.dart';
// // import 'package:firebase_signin/screens/home_screen.dart';
// import 'home_screen.dart';
// // import 'package:firebase_signin/screens/reset_password.dart';
// import 'reset_password.dart';
// // import 'package:firebase_signin/screens/signup_screen.dart';
// import 'signup_screen.dart';
// // import 'package:firebase_signin/utils/color_utils.dart';
// import 'package:flutter_application_1/utils/color_utils.dart';
// import 'package:flutter/material.dart';

// class SignInScreen extends StatefulWidget {
//   const SignInScreen({super.key});

//   @override
//   _SignInScreenState createState() => _SignInScreenState();
// }

// class _SignInScreenState extends State<SignInScreen> {
//   final TextEditingController _passwordTextController = TextEditingController();
//   final TextEditingController _emailTextController = TextEditingController();
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Container(
//         width: MediaQuery.of(context).size.width,
//         height: MediaQuery.of(context).size.height,
//         decoration: BoxDecoration(
//             gradient: LinearGradient(colors: [
//           hexStringToColor("CB2B93"),
//           hexStringToColor("9546C4"),
//           hexStringToColor("5E61F4")
//         ], begin: Alignment.topCenter, end: Alignment.bottomCenter)),
//         child: SingleChildScrollView(
//           child: Padding(
//             padding: EdgeInsets.fromLTRB(
//                 20, MediaQuery.of(context).size.height * 0.2, 20, 0),
//             child: Column(
//               children: <Widget>[
//                 logoWidget("assets/images/logo1.png"),
//                 const SizedBox(
//                   height: 30,
//                 ),
//                 reusableTextField("Enter UserName", Icons.person_outline, false,
//                     _emailTextController),
//                 const SizedBox(
//                   height: 20,
//                 ),
//                 reusableTextField("Enter Password", Icons.lock_outline, true,
//                     _passwordTextController),
//                 const SizedBox(
//                   height: 5,
//                 ),
//                 forgetPassword(context),
//                 firebaseUIButton(context, "Sign In", () {
//                   FirebaseAuth.instance
//                       .signInWithEmailAndPassword(
//                           email: _emailTextController.text,
//                           password: _passwordTextController.text)
//                       .then((value) {
//                     Navigator.push(context,
//                         MaterialPageRoute(builder: (context) => HomeScreen()));
//                   }).onError((error, stackTrace) {
//                     print("Error ${error.toString()}");
//                   });
//                 }),
//                 signUpOption()
//               ],
//             ),
//           ),
//         ),
//       ),
//     );
//   }

//   Row signUpOption() {
//     return Row(
//       mainAxisAlignment: MainAxisAlignment.center,
//       children: [
//         const Text("Don't have account?",
//             style: TextStyle(color: Colors.white70)),
//         GestureDetector(
//           onTap: () {
//             Navigator.push(context,
//                 MaterialPageRoute(builder: (context) => SignUpScreen()));
//           },
//           child: const Text(
//             " Sign Up",
//             style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
//           ),
//         )
//       ],
//     );
//   }

//   Widget forgetPassword(BuildContext context) {
//     return Container(
//       width: MediaQuery.of(context).size.width,
//       height: 35,
//       alignment: Alignment.bottomRight,
//       child: TextButton(
//         child: const Text(
//           "Forgot Password?",
//           style: TextStyle(color: Colors.white70),
//           textAlign: TextAlign.right,
//         ),
//         onPressed: () => Navigator.push(
//             context, MaterialPageRoute(builder: (context) => ResetPassword())),
//       ),
//     );
//   }
// }


// import 'package:firebase_auth/firebase_auth.dart';
// import 'package:flutter/material.dart';
// import 'home_screen.dart';
// import 'reset_password.dart';
// import 'signup_screen.dart';
// import 'package:flutter_application_1/utils/color_utils.dart';
// import 'package:flutter_application_1/reusable_widgets/reusable_widget.dart';

// class SignInScreen extends StatefulWidget {
//   const SignInScreen({super.key});

//   @override
//   _SignInScreenState createState() => _SignInScreenState();
// }

// class _SignInScreenState extends State<SignInScreen> {
//   final TextEditingController _passwordTextController = TextEditingController();
//   final TextEditingController _emailTextController = TextEditingController();
//   bool _keepMeLoggedIn = false; // For the "Keep me logged in" checkbox

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Container(
//         width: MediaQuery.of(context).size.width,
//         height: MediaQuery.of(context).size.height,
//         decoration: BoxDecoration(
//           color: hexStringToColor("1E90FF"), // Replace with your color
//         ),
//         child: SingleChildScrollView(
//           child: Padding(
//             padding: EdgeInsets.fromLTRB(
//                 20, MediaQuery.of(context).size.height * 0.15, 20, 0),
//             child: Column(
//               children: <Widget>[
//                 // Logo Widget
//                 logoWidget("assets/images/logo1.png"),
//                 const SizedBox(height: 30),

//                 // Welcome Back! Title
//                 const Text(
//                   "Welcome Back!",
//                   style: TextStyle(
//                     color: Colors.white,
//                     fontSize: 30,
//                     fontWeight: FontWeight.bold,
//                   ),
//                 ),
//                 const SizedBox(height: 10),
//                 const Text(
//                   "Login to your account",
//                   style: TextStyle(
//                     color: Colors.white70,
//                     fontSize: 16,
//                   ),
//                 ),
//                 const SizedBox(height: 30),

//                 // Email Address Field
//                 reusableTextField(
//                   "Email Address",
//                   Icons.email_outlined,
//                   false,
//                   _emailTextController,
//                 ),
//                 const SizedBox(height: 20),

//                 // Password Field
//                 reusableTextField(
//                   "Enter Password",
//                   Icons.lock_outline,
//                   true,
//                   _passwordTextController,
//                 ),
//                 const SizedBox(height: 10),

//                 // Keep me logged in Checkbox
//                 Row(
//                   children: [
//                     Checkbox(
//                       value: _keepMeLoggedIn,
//                       onChanged: (bool? value) {
//                         setState(() {
//                           _keepMeLoggedIn = value!;
//                         });
//                       },
//                       activeColor: Colors.white,
//                       checkColor: Colors.purple,
//                     ),
//                     const Text(
//                       "Keep me logged in",
//                       style: TextStyle(color: Colors.white70),
//                     ),
//                   ],
//                 ),
//                 const SizedBox(height: 10),

//                 // Forgot Password
//                 forgetPassword(context),
//                 const SizedBox(height: 20),

//                 // Login Button
//                 firebaseUIButton(context, "Login", () {
//                   FirebaseAuth.instance
//                       .signInWithEmailAndPassword(
//                     email: _emailTextController.text,
//                     password: _passwordTextController.text,
//                   )
//                       .then((value) {
//                     Navigator.push(
//                       context,
//                       MaterialPageRoute(builder: (context) => HomeScreen()),
//                     );
//                   }).onError((error, stackTrace) {
//                     print("Error ${error.toString()}");
//                   });
//                 }),
//                 const SizedBox(height: 20),

//                 // Sign Up Option
//                 signUpOption(),
//               ],
//             ),
//           ),
//         ),
//       ),
//     );
//   }

//   Row signUpOption() {
//     return Row(
//       mainAxisAlignment: MainAxisAlignment.center,
//       children: [
//         const Text(
//           "Don't have an account?",
//           style: TextStyle(color: Colors.white70),
//         ),
//         GestureDetector(
//           onTap: () {
//             Navigator.push(
//               context,
//               MaterialPageRoute(builder: (context) => SignUpScreen()),
//             );
//           },
//           child: const Text(
//             " Sign Up",
//             style: TextStyle(
//               color: Colors.white,
//               fontWeight: FontWeight.bold,
//             ),
//           ),
//         ),
//       ],
//     );
//   }

//   Widget forgetPassword(BuildContext context) {
//     return Container(
//       width: MediaQuery.of(context).size.width,
//       height: 35,
//       alignment: Alignment.bottomRight,
//       child: TextButton(
//         child: const Text(
//           "Forgot Password?",
//           style: TextStyle(color: Colors.white70),
//           textAlign: TextAlign.right,
//         ),
//         onPressed: () => Navigator.push(
//           context,
//           MaterialPageRoute(builder: (context) => ResetPassword()),
//         ),
//       ),
//     );
//   }
// }

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'home_screen.dart';
import 'reset_password.dart';
import 'signup_screen.dart';
import 'package:flutter_application_1/utils/color_utils.dart';
import 'package:flutter_application_1/reusable_widgets/reusable_widget.dart';

class SignInScreen extends StatefulWidget {
  const SignInScreen({super.key});

  @override
  _SignInScreenState createState() => _SignInScreenState();
}

class _SignInScreenState extends State<SignInScreen> {
  final TextEditingController _passwordTextController = TextEditingController();
  final TextEditingController _emailTextController = TextEditingController();
  bool _keepMeLoggedIn = false; // For the "Keep me logged in" checkbox

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
                    "Login Account",
                    style: TextStyle(
                      color: Color.fromARGB(255, 51, 85, 255),
                      fontWeight: FontWeight.w900,
                      fontFamily: "Roboto",
                      fontSize: 30,
                    ),
                  ),
                  SizedBox(height: 5),
                  Text(
                    "Welcome Back!",
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
              SizedBox(height: 20),
              Column(
              children: <Widget>[
                // Logo Widget
                logoWidget("assets/images/logo1.png"),
                const SizedBox(height: 30),

                // Welcome Back! Title
                const Text(
                  "Welcome Back!",
                  style: TextStyle(
                    color: Colors.black, // Black text for contrast
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 10),
                const Text(
                  "Login to your account",
                  style: TextStyle(
                    color: Colors.black54, // Dark gray text for contrast
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 30),

                // Email Address Field
                reusableTextField(
                  "Email Address",
                  Icons.email_outlined,
                  false,
                  _emailTextController,
                ),
                const SizedBox(height: 20),

                // Password Field
                reusableTextField(
                  "Enter Password",
                  Icons.lock_outline,
                  true,
                  _passwordTextController,
                ),
                const SizedBox(height: 10),

                // Keep me logged in Checkbox
                Row(
                  children: [
                    Checkbox(
                      value: _keepMeLoggedIn,
                      onChanged: (bool? value) {
                        setState(() {
                          _keepMeLoggedIn = value!;
                        });
                      },
                      activeColor: Color.fromARGB(255, 51, 85, 255), // Blue checkbox
                      checkColor: Colors.white, // White checkmark
                    ),
                    const Text(
                      "Keep me logged in",
                      style: TextStyle(color: Colors.black54), // Dark gray text
                    ),
                  ],
                ),
                const SizedBox(height: 10),

                // Forgot Password
                forgetPassword(context),
                const SizedBox(height: 10),

                // Login Button
                firebaseUIButton(context, "Login", () {
                  FirebaseAuth.instance
                      .signInWithEmailAndPassword(
                    email: _emailTextController.text,
                    password: _passwordTextController.text,
                  )
                      .then((value) {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => HomeScreen()),
                    );
                  }).onError((error, stackTrace) {
                    print("Error ${error.toString()}");
                  });
                }),
                const SizedBox(height: 5),

                // Sign Up Option
                signUpOption(),
                const SizedBox(height: 20),
              ])
            ],
          )
        ),
      ),
    );
  }

  Row signUpOption() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Text(
          "Don't have an account?",
          style: TextStyle(color: Colors.black54), // Dark gray text
        ),
        GestureDetector(
          onTap: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => SignUpScreen()),
            );
          },
          child: const Text(
            " Sign Up",
            style: TextStyle(
              color: Color.fromARGB(255, 51, 85, 255), // Blue text for the "Sign Up" link
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ],
    );
  }

  Widget forgetPassword(BuildContext context) {
    return Container(
      width: MediaQuery.of(context).size.width,
      height: 35,
      alignment: Alignment.bottomRight,
      child: TextButton(
        child: const Text(
          "Forgot Password?",
          style: TextStyle(color: Color.fromARGB(255, 51, 85, 255)), // Blue text for the link
          textAlign: TextAlign.right,
        ),
        onPressed: () => Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => ResetPassword()),
        ),
      ),
    );
  }
}