/**
 * Root Layout — BioDreamer Web Application.
 *
 * Purpose:
 *   The top-level layout component that wraps all pages. Provides:
 *     - Global CSS imports (Tailwind, custom styles)
 *     - HTML metadata (title, description, favicon)
 *     - Persistent layout elements: Navbar (top), Sidebar (left), Footer (bottom)
 *     - Global providers: theme (dark/light), API client context, toast notifications
 *
 * Structure:
 *   <html>
 *     <body>
 *       <Navbar />           — logo, module navigation, model status, dark mode toggle
 *       <div className="flex">
 *         <Sidebar />        — secondary navigation, recent jobs, quick links
 *         <main>{children}</main>  — page content
 *       </div>
 *       <Footer />           — version, links, HF Hub status
 *     </body>
 *   </html>
 */
